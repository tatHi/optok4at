import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools
import numpy as np
import sys
import unigramNLM
from multigram import mdp
from time import time

class OpTok(nn.Module):
    def __init__(self, mlm, 
                       lmEmbed,
                       m, 
                       n, 
                       topK, 
                       selectMode='sampling',
                       lam=0.2,
                       mTest=1,
                       nTest=1,
                       lamTest=1.0,
                       selectModeTest='top'):
        super().__init__()
        self.mlm = mlm
        self.lmEmbed = lmEmbed
        self.m = m
        self.n = n

        self.lam = lam
        self.selectMode = selectMode

        # set parameter for test
        self.mTest = mTest
        self.nTest = nTest
        self.lamTest = lamTest
        self.selectModeTest = selectModeTest
        ###

        vocabSize = len(mlm.vocab) #+ 1
        embedSize = self.lmEmbed.weight.shape[1]

        #self.unkCharIdx = vocabSize - 1 # last index, also used as padding in bilstm etc
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

        self.topK = topK

        self.CACHE_log_theta = None
        self.CACHE_log_smoothed_theta = None
        self.CACHE_vocab = None

    def encode(self,  xss, m, encoder):
        # encode
        xss_wo_pad = [[x for x in xs if x[0]!=self.minfPaddingIdx] for xs in xss]
        ls = [len(xs) for xs in xss_wo_pad]

        xss_wo_pad_flatten = [x for xs in xss_wo_pad for x in xs]

        yss, hss = encoder(xss_wo_pad_flatten, padding_idx=self.unkCharIdx)
        # TODO: padding_idx is confusing. change name as unk_idx
        # this proc may be related to the bilstm encoder?

        yss_pad = []
        hss_pad = []
        pointer = 0
        for l in ls:
            yss_pad.append(yss[pointer:pointer+l])
            hss_pad.append(hss[pointer:pointer+l])
            pointer += l

        yss_pad = rnn.pad_sequence(yss_pad, batch_first=True)

        if yss_pad.shape[1] < m:
            yss_pad = F.pad(yss_pad, (0, 0, 0, m-yss_pad.shape[1]))

        return yss_pad, hss_pad

    def __calcAttention(self, log_theta, idNbests, m):

        xs = [(idNth, len(idNth)) for idNbest in idNbests for idNth in idNbest]
        idNbests, lens = zip(*xs)
        maxL = max(lens)

        logPs = log_theta.unsqueeze(0)[:,[idNth + [self.zeroPaddingIdx]*(maxL-ln) for idNth, ln in zip(idNbests, lens)]] 
        logPs = torch.sum(logPs, dim=2)
        logPs = logPs.view(-1, m)

        attn = torch.exp(logPs - torch.logsumexp(logPs, dim=1, keepdim=True))

        return attn, logPs

    def __getLogTheta(self, lam, selectMode):
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
        elif selectMode=='top' or selectMode=='sampling':
            log_theta, selectedIds = self.nlm.getSelectedLogUnigramProbs(
                                                    self.lmEmbed, 
                                                    self.topK, 
                                                    mode=selectMode, 
                                                    lam=lam,
                                                    mustBeIncludeIdSet=self.mlm.getCharIdSet() | {self.unkCharIdx,})
            log_smoothed_theta = lam * log_theta
            vocab = set([self.mlm.id2word[i] for i in selectedIds])
        else:
            print('selectMode should be top, sampling or normal.'); exit()
        
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

    def __getNbests(self, lines, log_theta, m, n, lam, vocab=None, ffbsMode=False):
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
        
        idNbests = [mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar')
                        for idTable, logProbTable in zip(idTables, logProbTables)]

        # add pad if len(idNbest) < m
        idNbests = [idNbest + ([self.minfPaddingIdx],)*(m-len(idNbest)) for idNbest in idNbests]
        nbests = [[[self.mlm.id2word[i] if i in self.mlm.id2word else '[EXPAD]'
                    for i in inb] for inb in idNbest] for idNbest in idNbests]
        
        if ffbsMode and self.training:
            # represent as n-best tokenization where n=1
            idFFBSs = [[mdp.samplingIdSegmentation(idTable, logProbTable*lam)]
                                for idTable, logProbTable in zip(idTables, logProbTables)]
            ffbss = [[[self.mlm.id2word[i] if i in self.mlm.id2word else '[EXPAD]'
                        for i in inb] for inb in idFFBS] for idFFBS in idFFBSs]
        else:
            idFFBSs, ffbss = None, None

        return nbests, idNbests, ffbss, idFFBSs

    def __getUnigramLoss(self, nbests, logPs, attn):

        nonPadIdx = torch.where(logPs!=float('-inf'))
        weightedLogPs = - logPs[nonPadIdx] * attn.squeeze(1)[nonPadIdx] 
        lens_wo_pad = torch.tensor([len(nth) for nbest in nbests for nth in nbest if nth[0] != '[EXPAD]']).to(attn.device.type)

        uniLoss = torch.sum(weightedLogPs / lens_wo_pad) / weightedLogPs.shape[0] #len(nbests)
        return uniLoss

    def forward(self, lines):
        '''
        if you want to use different lam / selectMode for each iteration,
        change self.lam / self.selectMode directly.
        '''
        # TODO: implement setter of aboce hyperparameters
        # TODO: implement scheduler of lam

        # reset hyperparameters for inference
        n = self.n if self.training else self.nTest
        m = self.m if self.training else self.mTest
        lam = self.lam if self.training else self.lamTest
        selectMode = self.selectMode if self.training else self.selectModeTest
            
        #log_theta = self.__getLogTheta(lam)
        log_theta, log_smoothed_theta, vocab = self.__getLogTheta(lam, selectMode)

        #st = time()
        nbests, idNbests, ffbss, idFFBSs = self.__getNbests(lines, log_theta, m, n, lam, vocab, ffbsMode=True)

        # gumbel softmax
        attn, logPs = self.__calcAttention(log_theta, idNbests, m=m)

        # weighting
        attn = attn.view(len(lines), -1, m)

        # unigram loss
        uniLoss = self.__getUnigramLoss(nbests, logPs, attn.detach())

        return nbests, idNbests, ffbss, idFFBSs, attn, logPs, uniLoss

    def makeLogProbTable(self, idTable, theta):
        logProbTable = theta[idTable.flatten()]
        logProbTable = logProbTable.reshape(idTable.shape)
        return logProbTable

