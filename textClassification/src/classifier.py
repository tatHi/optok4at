import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import bilstmEncoder
import optok

class Classifier(nn.Module):
    def __init__(self, mlm, 
                       embedSize, 
                       hidSize, 
                       labelSize, 
                       m, 
                       n, 
                       topK, 
                       lam,
                       selectMode,
                       dropoutRate=0.5,
                       useIndividualEmbed=False):
        super().__init__()
        
        useIndividualEmbed = useIndividualEmbed

        # prepare embed
        lmEmbed = nn.Embedding(len(mlm.vocab), embedSize)
        if useIndividualEmbed:
            # if use individual embed is true, the model use different embedding 
            # for the language model and the classifier
            encEmbed = nn.Embedding(len(mlm.vocab), embedSize)
        else:
            encEmbed = lmEmbed

        # prepare encoder
        self.encoder = bilstmEncoder.BiLSTMEncoder(embedSize, hidSize, encEmbed)
        
        self.ot = optok.OpTok(mlm, 
                              lmEmbed=lmEmbed,
                              m=m,
                              n=n,
                              topK=topK,
                              selectMode=selectMode,
                              lam=lam,
                              )

        self.dropout = nn.Dropout(p=dropoutRate)
        self.linear = nn.Linear(hidSize, labelSize)
        
    def encode(self, xss):
        #xss_wo_pad = [[x for x in xs if x[0]!=self.ot.minfPaddingIdx] for xs in xss]
        xss_wo_pad = [[x if x[0]!=self.ot.minfPaddingIdx else [0] for x in xs] for xs in xss]
        xss_wo_pad_flatten = [x for xs in xss_wo_pad for x in xs]

        yss, hss = self.encoder(xss_wo_pad_flatten, padding_idx=self.ot.unkCharIdx)
        return yss, hss

    def calcScores(self, xss):
        vs, _ = self.encode(xss)
        vs = self.dropout(vs)
        scores = self.linear(vs)
        return scores

    def calcCLLoss(self, xss, labels):
        scores = self.calcScores(xss)
        loss = F.cross_entropy(scores, labels)
        return loss

    def calcOpTokLoss(self, idNbests, labels, attn):
        with torch.no_grad():
            scores = self.calcScores(idNbests)
        
        # repeat labels
        ls = [len([0 for x in idNbest]) for idNbest in idNbests]
        labelsIds = [label for l,label ,in zip(ls, labels) for i in range(l)]
        repeatIds = [i for i, l in enumerate(ls) for _ in range(l)]
        labels = labels[repeatIds]
        losses = F.cross_entropy(scores, labels, reduction='none') 

        loss_pad = []
        pointer = 0
        for l in ls:
            loss_pad.append(losses[pointer:pointer+l])
            pointer += l

        loss_pad = rnn.pad_sequence(loss_pad, batch_first=True)

        if loss_pad.shape[1] < self.ot.m:
            loss_pad = F.pad(loss_pad, (0, 0, 0, self.ot.m-loss_pad.shape[1])) 

        attn = attn.squeeze(1)
        otLoss = loss_pad * attn
        otLoss = (otLoss.sum().sum())/len(idNbests)

        return otLoss

    def forward(self, lines, labels=None):
        nbests, idNbests, ffbss, idFFBSs, attn, logPs, uniLoss = self.ot(lines)
        
        if labels is not None:
            clLoss = self.calcCLLoss(idFFBSs, labels)
            otLoss = self.calcOpTokLoss(idNbests, labels, attn)
            return clLoss, otLoss, uniLoss
        else:
            scores = self.calcScores(idNbests)
            return scores, uniLoss

    def forwardWithGivenSegmentation(self, xss):
        vs = self.ot.encode([xss], len(xss))[0]
        return self.calcScores(vs)

