import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools
import numpy as np
import sys
from . import unigramNLM
from multigram import mdp
from time import time

class OpTok(nn.Module):
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
                       samplingModeTest='top',
                       selectModeTest='normal',
                       lamTest=1.0,
                       tauTest=0.01):
        super().__init__()
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

        self.CACHE_log_smoothed_theta = None
        self.CACHE_log_theta = None
        self.CACHE_vocab = None

        self.bertMode = False

    def encode(self,  xss, m):
        # encode

        xss_wo_pad = [[x for x in xs if x[0]!=self.minfPaddingIdx] for xs in xss]
        ls = [len(xs) for xs in xss_wo_pad] # nbests

        eos = [self.mlm.word2id['</s>']]
        pad = [self.mlm.word2id['<pad>']]
        xss_wo_pad_flatten = [x + eos for xs in xss_wo_pad for x in xs]
        src_lengths = [len(xs) for xs in xss_wo_pad_flatten]
        maxL = max(src_lengths)
        xss_wo_pad_formated = [xs+pad*(maxL-l) for l,xs in zip(src_lengths, xss_wo_pad_flatten)]

        src_tokens = torch.LongTensor(xss_wo_pad_formated).cuda()
        src_lengths = torch.LongTensor(src_lengths).cuda()
        
        if self.training:
            idsForTransLoss = []
            idsForOpTokLoss = []
            idsForRecover = []
            c = 0
            size = len(ls)
            for i, l in enumerate(ls):
                idsForTransLoss.append(c)
                idsForOpTokLoss += list(range(c+1, c+l))
                idsForRecover += [i]+list(range(size+c-i, size+c-i+l-1))
                c += l
            
            # for translation loss
            encoder_out1, encoder_padding_mask1, encoder_embedding1, encoder_states1, src_tokens1, src_lengths1 \
                    = self.encoder(src_tokens=src_tokens[idsForTransLoss], src_lengths=src_lengths[idsForTransLoss])

            # for optok loss
            with torch.no_grad():
                encoder_out2, encoder_padding_mask2, encoder_embedding2, encoder_states2, src_tokens2, src_lengths2 \
                        = self.encoder(src_tokens=src_tokens[idsForOpTokLoss], src_lengths=src_lengths[idsForOpTokLoss])

            # marge outputs of encoder by stacking and slicing
            encoder_out = torch.cat([encoder_out1, encoder_out2], axis=1)[:, idsForRecover, :]
            encoder_padding_mask = torch.cat([encoder_padding_mask1, encoder_padding_mask2], axis=0)[idsForRecover,:]
            encoder_embedding = torch.cat([encoder_embedding1, encoder_embedding2], axis=0)[idsForRecover, :, :]        
            
            encoder_states = None
            src_tokens = None
            src_lengths = None
        else:
            encoder_out, encoder_padding_mask, encoder_embedding, encoder_states, src_tokens, src_lengths \
                    = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        
        # (T, B, H) -> (B, T, H)
        encoder_out = encoder_out.permute(1,0,2)

        encoder_out_pad = []
        encoder_padding_mask_pad = []
        pointer = 0
        for l in ls:
            encoder_out_pad.append(encoder_out[pointer:pointer+l])
            encoder_padding_mask_pad.append(encoder_padding_mask[pointer:pointer+l])
            pointer += l

        encoder_out_pad = rnn.pad_sequence(encoder_out_pad, batch_first=True)
        encoder_padding_mask_pad = rnn.pad_sequence(encoder_padding_mask_pad, batch_first=True, padding_value=False)

        if encoder_out.shape[1] < m:
            # when the maximum m is lower than specified m
            encoder_out_pad = F.pad(encoder_out_pad, (0, 0, 0, m-encoder_out_pad.shape[1]), value=0)
            encoder_padding_mask_pad = F.pad(encoder_padding_mask_pad, 
                                             (0, 0, 0, m-encoder_padding_mask_pad.shape[1]), 
                                             value=False)

        return encoder_out_pad, encoder_padding_mask_pad, encoder_embedding, encoder_states, src_tokens, src_lengths

    def __calcAttention(self, log_theta, idNbests, m, tau=None):

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

    def __getLogTheta(self, lam, selectMode='normal'):
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
            vocab = None
        else:
            log_theta, selectedIds = self.nlm.getSelectedLogUnigramProbs(
                                                    self.lmEmbed, 
                                                    self.topK, 
                                                    mode=selectMode, 
                                                    lam=lam,
                                                    mustBeIncludeIdSet=self.mlm.getCharIdSet() | {self.unkCharIdx,})
            vocab = set([self.mlm.id2word[i] for i in selectedIds])
        log_smoothed_theta = lam * log_theta

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
            self.CACHE_log_smoothed_theta = log_smoothed_theta
            self.CACHE_log_theta = log_theta
            self.CACHE_vocab = vocab
        return log_theta, log_smoothed_theta, vocab

    def __getNbests(self, lines, log_theta, m, n, lam=1.0, vocab=None):
        if vocab is None:
            vocab = self.mlm.vocab

        # nbests
        with torch.no_grad():
            log_theta = log_theta.cpu().detach().numpy().astype(float)
            idTables = [self.mlm.makeIdTable(
                            line,
                            paddingIdx=self.minfPaddingIdx,
                            unkCharIdx=self.unkCharIdx,
                            vocab=vocab
                        )  for line in lines]

            logProbTables = [self.makeLogProbTable(
                                idTable,
                                log_theta)
                             for idTable in idTables]

            idNbests = [mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar', lam=lam)
                            for idTable, logProbTable in zip(idTables, logProbTables)]
                                
            if self.ffbsMode and self.training:
                idNbests = [(mdp.samplingIdSegmentation(idTable, logProbTable*lam),) + idNbest
                            for idNbest, idTable, logProbTable in zip(idNbests, idTables, logProbTables)]
        
            # add pad for null sentence as "<unk>" to avoid -inf
            idNbests = [tuple(inb if inb else [self.mlm.word2id['<unk>']] for inb in idNbest) for idNbest in idNbests]

            # add pad if len(idNbest) < m
            k = m+1 if self.ffbsMode and self.training else m

            # use unk token
            idNbests = [idNbest + ([self.mlm.word2id['<unk>']],)*(k-len(idNbest)) for idNbest in idNbests]
                
            if self.bertMode:
                CLS = [self.mlm.word2id['[CLS]']]
                SEP = [self.mlm.word2id['[SEP]']]
                idNbests = [[CLS+inb+SEP for inb in idNbest] for idNbest in idNbests]
    
        nbests = [[[self.mlm.id2word[i] if i in self.mlm.id2word else '[EXPAD]'
                    for i in inb] if inb else ['[EXPAD]'] for inb in idNbest] for idNbest in idNbests]
        
        return nbests, idNbests

    def __getUnigramLoss(self, nbests, logPs, attn):
        nonPadIdx = torch.where(logPs!=float('-inf'))
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
    
    def forward(self, lines, src_tokens=None):
        '''
        if you want to use different lam / tau / selectMode for each iteration,
        change self.lam / self.tau / self.selectMode directly.
        '''
        # TODO: implement setter of aboce hyperparameters
        # TODO: implement scheduler of lam/tau

        # reset hyperparameters for inference
        n = self.n if self.training else self.nTest
        m = self.m if self.training else self.mTest
        tau = self.tau if self.training else self.tauTest
        lam = self.lam if self.training else self.lamTest
        selectMode = self.selectMode if self.training else self.selectModeTest
            
        log_theta, log_smoothed_theta, vocab = self.__getLogTheta(lam, selectMode)

        nbests, idNbests = self.__getNbests(lines, log_theta, m, n, lam, vocab)

        # gumbel softmax
        if self.ffbsMode and self.training:
            # use [1:]'s attention because 0th content of idNbest is FFBS tokenization when FFBS mode
            attn, softattn, logPs = self.__calcAttention(
                                        log_theta, 
                                        [idNbest[1:] for idNbest in idNbests], 
                                        m=m, 
                                        tau=tau)
        else:
            attn, softattn, logPs = self.__calcAttention(log_theta, idNbests, m=m, tau=tau)



        # encodes
        encoder_out_pad, encoder_padding_mask_pad, encoder_embedding, encoder_states, src_tokens, src_lengths \
            = self.encode(idNbests, m=m+1 if self.ffbsMode and self.training else m)

        B, M, T, H = encoder_out_pad.shape
        encoder_out_pad = encoder_out_pad.view(B*M,T,H)
        encoder_padding_mask_pad = encoder_padding_mask_pad.view(B*M, T)

        # reshape for encoder out
        encoder_out_pad = encoder_out_pad.permute(1,0,2) # (BM,T,H) -> (T, BM, H)

        # unigram loss
        if self.training:
            if self.ffbsMode:
                uniLoss = self.__getUnigramLoss([nbest[1:] for nbest in nbests], logPs, attn.detach())
            else:
                uniLoss = self.__getUnigramLoss(nbests, logPs, attn.detach())
        else:
            uniLoss = None

        return encoder_out_pad, encoder_padding_mask_pad, encoder_embedding, encoder_states, src_tokens, src_lengths, attn, uniLoss

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
