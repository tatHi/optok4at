import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools
import numpy as np
import sys
from . import unigramNLM
from multigram import mdp
from multigram import tokenizer as T
from time import time
from typing import Dict, List, NamedTuple, Optional
from torch import Tensor

checkSpeed=False

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("src_optok_probs", Optional[Tensor])  # B x M; probabilities of tokenization calculated by optokEnc
    ],
)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class OpTokGen(nn.Module):
    def __init__(self, mlm, 
                       lmEmbed,
                       encoder, 
                       m, 
                       n, 
                       topK, 
                       samplingMode, 
                       ffbsMode,
                       selectMode='sampling',
                       lam=0.2,
                       tau=0.1,
                       mTest=1,
                       nTest=1,
                       samplingModeTest='soft',
                       selectModeTest='normal',
                       lamTest=1.0,
                       tauTest=0.01,
                       normalOpTokEnc=False,
                       normalOpTokDec=False):
        super().__init__()

        self.normalOpTokEnc = normalOpTokEnc
        self.normalOpTokDec = normalOpTokDec

        self.mlm = mlm
        self.lmEmbed = lmEmbed
        self.encoder = encoder
        self.m = m
        self.n = n

        self.lam = lam
        self.tau = tau
        self.selectMode = selectMode

        # set parameter for test
        self.mTest = mTest
        self.nTest = nTest
        self.samplingModeTest = samplingModeTest
        self.selectModeTest = selectModeTest
        self.lamTest = lamTest
        self.tauTest = tauTest

        vocabSize = len(mlm.vocab) #+ 1
        embedSize = self.lmEmbed.weight.shape[1]

        if '<unk>' in mlm.vocab:
            self.unkCharIdx = mlm.piece_to_id('<unk>') 
        elif '[UNK]' in mlm.vocab:
            self.unkCharIdx = mlm.word2id['[UNK]']
        else:
            # when using model dumped by older version, the model does not have unk token.
            # we use the last index as unkidx at that time.
            vocabSize += 1
            self.unkCharIdx = vocabSize-1
        
        self.minfPaddingIdx = vocabSize
        self.zeroPaddingIdx = vocabSize+1

        self.nlm = unigramNLM.UnigramNLM(vocabSize, embedSize, unkIdx=self.unkCharIdx)

        self.samplingMode = samplingMode

        self.topK = topK

        self.ffbsMode = ffbsMode

        self.CACHE_log_theta = None
        self.CACHE_log_smoothed_theta = None
        self.CACHE_vocab = None

        self.bertMode = False

    def calcLoss(self, yss, tss, ls, m):
        lprobs = F.log_softmax(yss.view(-1, len(self.mlm.vocab)))
        smoothedLoss, NLLLoss  = label_smoothed_nll_loss(
                                    lprobs,
                                    tss.view(tss.shape[0]*tss.shape[1]),
                                    epsilon=0.1,
                                    ignore_index=self.mlm.word2id['<pad>'],
                                    reduce=False)
        loss = smoothedLoss.flatten()

        loss = loss.view(tss.shape[0]//m, m, -1)
        loss = loss.sum(dim=2)
        return loss

    def encode(self,  xss, 
                      m,
                      encoder_out,
                      incremental_state,
                      features_only,
                      alignment_layer,
                      alignment_heads,
                      src_lengths,
                      return_all_hiddens):
        # encode
        
        # flatten with adding EOS
        # for fairseq, </s> is used as EOS
        EOS = self.mlm.word2id['</s>']
        PAD = self.mlm.word2id['<pad>']
        xss = [[EOS]+x+[EOS] for xs in xss for x in xs]

        # replace minfPaddingIdx with PAD
        xss = [[PAD if x==self.minfPaddingIdx else x for x in xs] for xs in xss]

        # padding
        ls = [len(xs)-1 for xs in xss] 
        maxL = max(ls)
        tss = [xs[1:]+[PAD]*(maxL-(len(xs)-1)) for xs in xss] # gold signal to decoder [a,b,c,</s>]
        xss = [xs[:-1]+[PAD]*(maxL-(len(xs)-1)) for xs in xss] # input to Decoder [</s>,a,b,c]

        # to cuda
        xss = torch.LongTensor(xss).cuda()
        tss = torch.LongTensor(tss).cuda()

        yss = self.encoder(
                    xss,
                    encoder_out,
                    incremental_state,
                    features_only,
                    alignment_layer,
                    alignment_heads,
                    src_lengths,
                    return_all_hiddens,
                    opTokM=None)[0] 
        loss = self.calcLoss(yss, tss, ls, m)

        return loss

    def __calcAttention(self, log_theta, idNbests, m, lam, tau=None):

        if self.bertMode:
            idNbests = [[inb[1:-1] for inb in idNbest] for idNbest in idNbests]

        xs = [(idNth, len(idNth)) for idNbest in idNbests for idNth in idNbest]
        idNbests, lens = zip(*xs)
        maxL = max(lens)

        logPs = log_theta.unsqueeze(0)[:,[idNth + [self.zeroPaddingIdx]*(maxL-ln) for idNth, ln in zip(idNbests, lens)]] 

        logPs = torch.sum(logPs, dim=2)
        logPs = logPs.view(-1, m)

        softattn = torch.exp(logPs - torch.logsumexp(logPs, dim=1, keepdim=True))

        if self.samplingMode == 'soft':
            attn = softattn
        elif self.samplingMode == 'gumbel':
            attn = F.gumbel_softmax(logPs, tau=tau, dim=1)
        elif self.samplingMode =='temp':
            logPs_norm = logPs - torch.max(logPs, dim=1, keepdim=True)[0]
            ps_temp = torch.exp(logPs_norm/tau)
            attn = ps_temp / torch.sum(ps_temp, dim=1, keepdim=True)

        return attn, softattn, logPs

    def getLogTheta(self, lam, selectMode='normal'):
        # cache
        if self.training:
            self.CACHE_log_theta = None
            self.CACHE_log_smoothed_theta = None
            self.CACHE_vocab = None
        else:
            if self.CACHE_log_smoothed_theta is not None:
                return self.CACHE_log_theta, self.CACHE_log_smoothed_theta, self.CACHE_vocab

        if selectMode=='normal':
            log_theta = self.nlm.getLogUnigramProbs(self.lmEmbed)
            log_smoothed_theta = lam * log_theta
            vocab = None
        else:
            log_theta, selectedIds = self.nlm.getSelectedLogUnigramProbs(
                                                    self.lmEmbed, 
                                                    self.topK, 
                                                    mode=selectMode, 
                                                    lam=lam,
                                                    mustBeIncludeIdSet=self.mlm.getCharIdSet() | {self.unkCharIdx,})
            log_smoothed_theta = lam * log_theta
            vocab = set([self.mlm.id2word[i] for i in selectedIds])
        
        # minf padding
        log_theta = F.pad(log_theta,
                          pad=(0,1),
                          value=float('-inf'))
        log_smoothed_theta = F.pad(log_smoothed_theta,
                          pad=(0,1),
                          value=float('-inf'))
        
        # zero padding
        log_theta = F.pad(log_theta,
                          pad=(0,1),
                          value=0)
        log_smoothed_theta = F.pad(log_smoothed_theta,
                          pad=(0,1),
                          value=0)

        # cache
        if not self.training and self.CACHE_log_smoothed_theta is None:
            self.CACHE_log_theta = log_theta
            self.CACHE_log_smoothed_theta = log_smoothed_theta
            self.CACHE_vocab = vocab
        return log_theta, log_smoothed_theta, vocab

    def __getNbests(self, lines, log_theta, m, n, lam=1.0, vocab=None):
        if vocab is None:
            vocab = self.mlm.vocab

        # nbests
        with torch.no_grad():
            st = time()
            log_theta = log_theta.cpu().detach().numpy().astype(float)
            if checkSpeed: print('>to numpy', time()-st)
            
            st = time()
            idTables = [self.mlm.makeIdTable(
                            line,
                            paddingIdx=self.minfPaddingIdx,
                            unkCharIdx=self.unkCharIdx,
                            vocab=vocab
                        )  for line in lines]
            if checkSpeed: print('>idTable', time()-st)

            st = time()
            logProbTables = [self.makeLogProbTable(
                                idTable,
                                log_theta)
                             for idTable in idTables]
            if checkSpeed: print('>logProbTable', time()-st)
            
            idNbests = [mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar', lam=lam)
                            for idTable, logProbTable in zip(idTables, logProbTables)]

            if self.ffbsMode and self.training:
                idNbests = [(mdp.samplingIdSegmentation(idTable, logProbTable*lam),) + idNbest
                            for idNbest, idTable, logProbTable in zip(idNbests, idTables, logProbTables)]
                    
            idNbests = [tuple(inb if inb else [self.mlm.word2id['<unk>']] for inb in idNbest) for idNbest in idNbests]
        
            # add pad if len(idNbest) < m
            k = m+1 if self.ffbsMode and self.training else m
            idNbests = [idNbest + ([self.minfPaddingIdx],)*(k-len(idNbest)) for idNbest in idNbests]

            if self.bertMode:
                CLS = [self.mlm.word2id['[CLS]']]
                SEP = [self.mlm.word2id['[SEP]']]
                idNbests = [[CLS+inb+SEP for inb in idNbest] for idNbest in idNbests]
        

        nbests = [[[self.mlm.id2word[i] if i in self.mlm.id2word else '[EXPAD]'
                    for i in inb] if inb else ['[EXPAD]'] for inb in idNbest] for idNbest in idNbests]
        
        return nbests, idNbests

    def __getUnigramLoss(self, nbests, logPs, attn):
        nonPadBool = logPs!=float('-inf')
        nonPadIdx = torch.where(nonPadBool)
        weightedLogPs = - logPs[nonPadIdx] * attn.squeeze(1)[nonPadIdx] 
        lens_wo_pad = torch.tensor([len(nth) for nbest in nbests for nth in nbest if nth[0] != '[EXPAD]']).to(attn.device.type)
        uniLoss = torch.sum(weightedLogPs / lens_wo_pad) / weightedLogPs.shape[0] #len(nbests)
        return uniLoss

    def __check(self, example, lines, attn, softattn, idNbests):
        if example not in lines:
            return
        
        idx = lines.index(example)
        for sa, a, inb in zip(softattn[idx], attn[idx], idNbests[idx]):
            print('%.5f\t%.5f'%(sa, a)+'\t'+'/'.join(self.mlm.id2word[w] 
                  if w != self.minfPaddingIdx and w != self.unkCharIdx else 'xxxxx' for w in inb))
        print('')
    

    def forwardForDecoder(
                        self,
                        nbests,
                        idNbests,
                        log_theta,
                        n,
                        m,
                        lam,
                        tau,
                        encoder_out,
                        incremental_state,
                        features_only,
                        alignment_layer,
                        alignment_heads,
                        src_lengths,
                        return_all_hiddens):

        encoderM = 1 if self.normalOpTokEnc else m
        if self.ffbsMode and self.training:
            if not self.normalOpTokEnc:
                encoderM += 1

            idNbests = [idNbest[1:] for idNbest in idNbests]
            if self.training:
                nbests = [nbest[1:] for nbest in nbests]

        with torch.no_grad():
            firstIds = list(range(0, encoder_out.encoder_padding_mask.shape[0], encoderM))

            mtimes_encoder_out = encoder_out.encoder_out.transpose(0,1)
            mtimes_encoder_out = mtimes_encoder_out[firstIds]
            tmp_encoder_padding_mask = encoder_out.encoder_padding_mask[firstIds]
            
            mtimes_encoder_out = mtimes_encoder_out.repeat(1,m,1).view((-1,)+mtimes_encoder_out.shape[1:])
            mtimes_encoder_out = mtimes_encoder_out.transpose(0,1)

            mtimes_encoder_padding_mask = \
                tmp_encoder_padding_mask.repeat(1,m).view(-1, tmp_encoder_padding_mask.shape[1])
            mtimes_encoder_embedding = encoder_out.encoder_embedding.repeat(1,m,1).view(
                                            (-1,)+encoder_out.encoder_embedding.shape[1:])

            mtimes_encoder_out = EncoderOut(
                encoder_out = mtimes_encoder_out,
                encoder_padding_mask = mtimes_encoder_padding_mask,
                encoder_embedding = mtimes_encoder_embedding,
                encoder_states = encoder_out.encoder_states,
                src_tokens = encoder_out.src_tokens,
                src_lengths = encoder_out.src_lengths,
                src_optok_probs = None
            )

            # encodes
            eachLoss = self.encode(
                                idNbests,
                                m,
                                mtimes_encoder_out,
                                incremental_state,
                                features_only,
                                alignment_layer,
                                alignment_heads,
                                src_lengths,
                                return_all_hiddens)

        
        if (log_theta is None) or (n==1 and m==1):
            # if n==1 and m==1, ignore attention.
            attn, softattn, logPs = None, None, None
            genLoss = eachLoss.flatten()
        else: 
            # calc attention
            attn, softattn, logPs = self.__calcAttention(log_theta, idNbests, m=m, lam=lam, tau=tau)

            # weighting
            genLoss = attn * eachLoss.detach()
            
        genLoss = genLoss.sum().sum()

        # unigram loss
        st = time()
        if self.training and not self.normalOpTokDec and 1<self.m:
            uniLoss = self.__getUnigramLoss(nbests, logPs, attn.detach())
        else:
            uniLoss = None
        if checkSpeed: print('uniLoss', time()-st)

        return genLoss, uniLoss

    def forwardForEncoder(
                        self,
                        idNbests,
                        n,
                        m,
                        lam,
                        tau,
                        encoder_out,
                        incremental_state,
                        features_only,
                        alignment_layer,
                        alignment_heads,
                        src_lengths,
                        return_all_hiddens):
        idNbests = [(idNbest[0],)*m for idNbest in idNbests] 

        if self.ffbsMode and self.training:
            mp1 = m+1
            ids = [i for i in range(len(idNbests)*mp1) if i%mp1!=0]
            
            encoder_out = EncoderOut(
                encoder_out = encoder_out.encoder_out[:,ids,:],
                encoder_padding_mask = encoder_out.encoder_padding_mask[ids,:],
                encoder_embedding = encoder_out.encoder_embedding[ids,:,:],
                encoder_states = encoder_out.encoder_states,
                src_tokens = encoder_out.src_tokens,
                src_lengths = encoder_out.src_lengths,
                src_optok_probs = encoder_out.src_optok_probs
            ) 


        # encodes
        with torch.no_grad():
            eachLoss = self.encode(
                                idNbests,
                                m,
                                encoder_out,
                                incremental_state,
                                features_only,
                                alignment_layer,
                                alignment_heads,
                                src_lengths,
                                return_all_hiddens)

        # weighting
        genLoss = eachLoss * encoder_out.src_optok_probs
        genLoss = genLoss.sum().sum()

        return genLoss

    def forwardForTranslation(
                        self,
                        idNbests,
                        n,
                        m,
                        lam,
                        tau,
                        encoder_out,
                        incremental_state,
                        features_only,
                        alignment_layer,
                        alignment_heads,
                        src_lengths,
                        return_all_hiddens):

        if self.normalOpTokEnc:
            m = 1
        if self.ffbsMode and self.training and not self.normalOpTokEnc:
            m += 1

        firstIds = list(range(0, encoder_out.encoder_padding_mask.shape[0], m))

        encoder_out = EncoderOut(
            encoder_out = encoder_out.encoder_out[:,firstIds],
            encoder_padding_mask = encoder_out.encoder_padding_mask[firstIds],
            encoder_embedding = encoder_out.encoder_embedding,#[firstIds],
            encoder_states = encoder_out.encoder_states,
            src_tokens = encoder_out.src_tokens,
            src_lengths = encoder_out.src_lengths,
            src_optok_probs = None
        )

        # decoder
        idNbests = [(idNbest[0],) for idNbest in idNbests]

        # encodes
        eachLoss = self.encode(
                            idNbests,
                            1, # m
                            encoder_out,
                            incremental_state,
                            features_only,
                            alignment_layer,
                            alignment_heads,
                            src_lengths,
                            return_all_hiddens)

        loss = eachLoss.sum().sum()

        return loss

    def forward(self, 
                target,
                lines,
                encoder_out,
                incremental_state,
                features_only,
                alignment_layer,
                alignment_heads,
                src_lengths,
                return_all_hiddens):

        # reset hyperparameters for inference
        n = self.n if self.training else self.nTest
        m = self.m if self.training else self.mTest
        tau = self.tau if self.training else self.tauTest
        lam = self.lam if self.training else self.lamTest
        selectMode = self.selectMode if self.training else self.selectModeTest
            
        if self.normalOpTokDec or not self.training:
            nbests, idNbests = None, [(pot.cpu().numpy().tolist()[:-1],)*m for pot in target]
            log_theta, log_smoothed_theta, vocab = None, None, None
        else:
            st = time()
            log_theta, log_smoothed_theta, vocab = self.getLogTheta(lam, selectMode)
            if checkSpeed: print('getLogTheta', time()-st)

            st = time()
            nbests, idNbests = self.__getNbests(lines, log_theta, m, n, lam, vocab)
            if checkSpeed: print('getNbests', time()-st)

        transLoss = self.forwardForTranslation(
                            idNbests,
                            n,
                            m,
                            lam,
                            tau,
                            encoder_out,
                            incremental_state,
                            features_only,
                            alignment_layer,
                            alignment_heads,
                            src_lengths,
                            return_all_hiddens)


        if self.normalOpTokEnc:
            encLoss = 0
        else:
            encLoss = self.forwardForEncoder(
                            idNbests,
                            n,
                            m,
                            lam,
                            tau,
                            encoder_out,
                            incremental_state,
                            features_only,
                            alignment_layer,
                            alignment_heads,
                            src_lengths,
                            return_all_hiddens)

        if self.normalOpTokDec:
            decLoss, uniLoss = 0, None
        else:
            decLoss, uniLoss = self.forwardForDecoder(
                            nbests,
                            idNbests,
                            log_theta,
                            n,
                            m,
                            lam,
                            tau,
                            encoder_out,
                            incremental_state,
                            features_only,
                            alignment_layer,
                            alignment_heads,
                            src_lengths,
                            return_all_hiddens)

        genLoss = encLoss + decLoss + transLoss

        return genLoss, uniLoss

    def forwardEncoder(self, lines):
        # forward only for encoder, as freezing unigram language model
        with torch.no_grad():
            log_theta, log_smoothed_theta, vocab = self.getLogTheta(self.lam, self.selectMode)
            nbests, idNbests = self.__getNbests(lines=lines, log_theta=log_smoothed_theta, m=1, n=1, vocab=vocab)
        eachLoss = self.encode(idNbests, m=self.m)
        genLoss = eachLoss.flatten()
        return genLoss

    def forwardULM(self, lines):
        # forward only for ulm, as freezing encoder
        log_theta, log_smoothed_theta, vocab = self.getLogTheta(self.lam, self.selectMode)
        with torch.no_grad():
            nbests, idNbests = self.__getNbests(lines=lines, log_theta=log_smoothed_theta, m=self.m, n=self.n, vocab=vocab)
            eachLoss = self.encode(idNbests, m=self.m)
        attn, softattn, logPs = self.__calcAttention(log_theta, idNbests, m=self.m, lam=self.lam, tau=self.tau)
        genLoss = attn * eachLoss
        genLoss = torch.sum(genLoss, dim=1)

        if 1<self.n and 1<self.m:
            uniLoss = self.__getUnigramLoss(nbests, logPs, attn.detach())
        else:
            uniLoss = None
       
        return genLoss, uniLoss

    def makeLogProbTable(self, idTable, theta):
        logProbTable = theta[idTable.flatten()]
        logProbTable = logProbTable.reshape(idTable.shape)
        return logProbTable

    def fitNLM2Theta(self):
        self.nlm.fitTo(self.lmEmbed, self.mlm.theta)

    def saveNLMasMLM(self, path):
        # make unigram dict
        theta = self.nlm.getUnigramProbs(self.lmEmbed).cpu().detach().numpy()
        self.mlm.theta = theta
        self.mlm.save(path)
        print('>>> DUMP LEARNED LM AS MLM')
