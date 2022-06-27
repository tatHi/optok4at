import os
import yaml
import argparse
import torch
import torch.nn as nn
import pickle
import numpy as np
import classifier
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import sentencepiece as spm
from multigram import lm
from multigram import util
from multigram import *
from time import time
import sys
import copy
import unicodedata

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLossHistory = []
validHistory = []
evalHistory = []

def train(data, labels, mlm, cl, opt, args):
    labels['train'] = torch.LongTensor(labels['train']).to(device)

    def freezeCLEmbedding(cl):
        print('freeze embedding for classifier')
        for p in cl.encoder.embed.parameters():
            p.requires_grad = False
        return cl
    def freezeOTEmbedding(cl):
        print('freeze embedding for optok')
        for p in cl.ot.lmEmbed.parameters():
            p.requires_grad = False
        return cl

    if args.freezeCLEmbedding:
        cl = freezeCLEmbedding(cl)
    if args.freezeOTEmbedding:
        cl = freezeOTEmbedding(cl)

    uniLossWeight = args.uniLossWeight

    ### MAIN TRAINING ###
    for epoch in range(args.maxEpoch):
        print('epoch: %d/%d'%(epoch+1, args.maxEpoch))

        indices = np.random.permutation(len(data['train']))
        batches = pack(indices, args.batchSize)

        accCLLoss = 0
        accOTLoss = 0
        accLMLoss = 0
        for batch in tqdm(batches):
            opt.zero_grad()
            xs = [data['train'][b] for b in batch]

            clLoss, otLoss, uniLoss = cl.forward(lines=xs, labels=labels['train'][batch])
            loss = clLoss + otLoss + uniLossWeight*uniLoss

            loss.backward()         
            opt.step()

            accCLLoss += clLoss.data
            accOTLoss += otLoss.data
            accLMLoss += uniLoss.data

        ### EVAL ###
        v_ps, e_ps = evaluate(cl, data, labels, args)
        if not args.debug and (len(validHistory)==1 or max(validHistory[:-1]) < validHistory[-1]):
            # dump predictions
            f = open('../results/%s/bestpreds.pickle'%(args.dirName), 'wb')
            pickle.dump({'valid':v_ps, 'eval':e_ps}, f)

            # dump best models
            saveNLMasMLM(cl, mlm, '../results/%s/mlm.best.pickle'%args.dirName)
            cl.to('cpu')
            torch.save(cl.state_dict(), '../results/%s/cl.best.model'%args.dirName)
            cl.to(device)

        ### EVAL ###

        accCLLoss = accCLLoss.data.tolist()
        accOTLoss = accOTLoss.data.tolist()
        accLMLoss = accLMLoss.data.tolist()
        print('accCLLoss:', accCLLoss)
        print('accOTLoss:', accOTLoss)
        print('accLMLoss:', accLMLoss)
        trainLossHistory.append(accCLLoss)
        
        showWordRank(cl, mlm, 10)

        if not args.debug and args.dumpEvalEachEpoch:
            dumpHistory('../results/%s/tmphist.txt'%args.dirName)

    return cl

def inference(cl, lines, args):
    indices = list(range(len(lines)))
    batches = pack(indices, args.batchSize)
    ps = []
    for batch in batches:
        xs = [lines[b] for b in batch]
        scores, _ = cl.forward(lines=xs)
        ps += torch.argmax(scores, dim=1).cpu().tolist()
    return ps

def evaluate(cl, data, labels, args):
    cl.eval()
    with torch.no_grad():
        v_ps = None
        e_ps = None
        if 'valid' in labels:
            v_ps = inference(cl, data['valid'], args)
            validHistory.append(f1_score(labels['valid'], v_ps, average='weighted'))
        if 'eval' in labels:
            e_ps = inference(cl, data['eval'], args)
            evalHistory.append(f1_score(labels['eval'], e_ps, average='weighted'))
        print('valid/test:', validHistory[-1], evalHistory[-1])
    cl.train()

    return v_ps, e_ps

def showWordRank(cl, mlm, topK):
    theta = cl.ot.nlm.getUnigramProbs(cl.ot.lmEmbed).cpu().data.numpy()
    wordScoreDict = {w:theta[i] for i,w in mlm.id2word.items()}
    for w,s in sorted(wordScoreDict.items(), key=lambda x:x[1], reverse=True)[:topK]:
        print(w,s)

def optTokenize(data, cl, mlm):
    # TODO: revise for wordpiece mode
    # TODO: modify not to use ndp, but mdp
    # inference
    theta = cl.ot.nlm.getLogUnigramProbs()
    idTables = [mlm.makeIdTable(line, paddingIdx=len(mlm.word2id)) for line in data]
    segData = [dp.viterbiSegmentation(line, multokEncoder.makeLogProbTable(idTable, theta))
               for line,idTable in tqdm(zip(data,idTables))]

    segFlat = [w for line in segData for w in line]
    loglikelihood = sum([theta[mlm.word2id[w]] for w in segFlat])/len(segFlat)
    print('log-likelihood:', loglikelihood.data.tolist())

    # dump
    path = 'tmpSeg-%s.txt'%util.getTimeStamp()
    with open(path, 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')
    print('>>> DUMP RESULTS')
    print('to '+path)

def saveNLMasMLM(cl, mlm, path):
    # make unigram dict
    theta = cl.ot.nlm.getUnigramProbs(cl.ot.lmEmbed).cpu().data.tolist()
    unigramDict = {w:theta[i] for i,w in mlm.id2word.items()}# if 0.<theta[i]}
    print('size of unigram dict:', len(unigramDict))

    # make new mlm
    neomlm = lm.MultigramLM()
    neomlm.setVocabFromUnigramDict(unigramDict, word2id=mlm.word2id, char2id=mlm.char2id)
    neomlm.save(path)
    print('>>> DUMP LEARNED LM AS MLM')

def pack(arr, size):
    batch = [arr[i:i+size] for i in range(0, len(arr), size)]
    return batch

def prepare(args):
    # data and labels
    data = {'train':[line.strip() for line in open(args.trainText)]}
    labels = {'train':[int(line.strip()) for line in open(args.trainLabel)]}

    if args.validText:
        data['valid'] = [line.strip() for line in open(args.validText)]
        labels['valid'] = [int(line.strip()) for line in open(args.validLabel)]
    if args.validText:
        data['eval'] = [line.strip() for line in open(args.evalText)]
        labels['eval'] = [int(line.strip()) for line in open(args.evalLabel)]

    # debug truncate
    if args.debug:
        for ty in data:
            data[ty] = data[ty][:1000]
        for ty in labels:
            labels[ty] = labels[ty][:1000]
    
    # check data size
    print('port\ttext\tlabel')
    for ty in data:
        print('%s:\t%d\t%d'%(ty, len(data[ty]), len(labels[ty])))

    # normalize as NFKC, full-width to half-width
    data = {k: [unicodedata.normalize('NFKC', v) for v in vs] for k, vs in data.items()}

    # language model
    if args.multigramLanguageModel and args.sentencePieceModel:
        print('Both MultigramLanguageModel and SentencePieceModel are specified.')
        print('You can specify only one model for the language model. -> EXIT')
        print('MLM:', args.multigramLanguageModel)
        print('SPM:', args.sentencePieceModel)
        exit()

    if args.sentencePieceModel:
        mlm = lm.MultigramLM()
        mlm.loadSentencePieceModel(args.multigramLanguageModel)
        data = {key:['▁'+line.replace(' ', '▁').replace('　','▁') for line in data[key]]
                for key in data}
        print('>>> LOAD SENTENCEPIECE MODEL')
    elif args.multigramLanguageModel:
        mlm = lm.MultigramLM()
        mlm.load(args.multigramLanguageModel)

        # add unk
        if '<unk>' not in mlm.vocab:
            mlm.addWordToVocab('<unk>', p=1e-7)
        
        print('>>> LOAD MULTIGRAM LM')
    else:
        # if no lm is specified, create new multigram language model using training split
        mlm = lm.MultigramLM(maxLength=8, minFreq=50)
        mlm.buildVocab(data=data['train'])
        print('>>> CREATE MULTIGRAM LM FROM TRAINING SPLIT')

    # classifier
    labelSize = len(set(labels['train']))
    cl = classifier.Classifier(mlm, 
                               embedSize=64, 
                               hidSize=256, 
                               labelSize=labelSize,
                               m=args.mSample,
                               n=args.nbest,
                               topK=args.selectSize,
                               lam=args.lam,
                               selectMode=args.selectMode,
                               dropoutRate=args.dropoutRate,
                               useIndividualEmbed=args.useIndividualEmbed)

    if args.pretrainedEmbed:
        print('>>> LOAD PRETRAINED EMBEDDING')
        cl.ot.lmEmbed.load_state_dict(torch.load(args.pretrainedEmbed))
        if args.useIndividualEmbed:
            cl.encoder.embed.load_state_dict(torch.load(args.pretrainedEmbed))

    cl.to(device)

    if args.debug:
        cl.ot.nlm.fitTo(cl.ot.lmEmbed, mlm.theta, maxEpoch=10)
    else:
        cl.ot.nlm.fitTo(cl.ot.lmEmbed, mlm.theta)

    # optimizer
    opt = torch.optim.Adam(cl.parameters())

    return data, labels, mlm, cl, opt

def dumpHistory(path):
    print('>>> SHOW RESULTS')
    with open(path, 'w') as f:
        print('LOSS VALID TEST')
        f.write('LOSS VALID TEST\n')
        bestV = 0
        bestE = 0
        for l,v,e in zip(trainLossHistory, validHistory, evalHistory):
            print(l,v,e)
            f.write('%f %f %f\n'%(l,v,e))
            if bestV < v:
                bestV = v
                bestE = e
        print('best:', bestE)
        f.write('best: %f'%bestE)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tt', '--trainText')
    parser.add_argument('-tl', '--trainLabel')
    parser.add_argument('-vt', '--validText', default=None)
    parser.add_argument('-vl', '--validLabel', default=None)
    parser.add_argument('-et', '--evalText', default=None) 
    parser.add_argument('-el', '--evalLabel', default=None)   
    parser.add_argument('-mlm', '--multigramLanguageModel', default=None, 
                        help='multigram language model for tokenization')
    parser.add_argument('-spm', '--sentencePieceModel', default=None, 
                        help='sentencepiece model for tokenization')
    parser.add_argument('-me', '--maxEpoch', default=20, type=int)
    parser.add_argument('-bs', '--batchSize', default=128, type=int)
    parser.add_argument('-ulw', '--uniLossWeight', default=0.01, type=float,
                        help='weight for loss value to maintain unigram language model introduced in Hiraoka2020')
    parser.add_argument('-l', '--lam', default=0.2, type=float,
                        help='smoothing coefficient for a distribution to sample restricted vocabulary introduced in Hiraoka2020')
    parser.add_argument('-nb', '--nbest', default=5, type=int)
    parser.add_argument('--debug', action='store_true',
                        help='if true, the size of corpus and the pre-training epoch of the initized neural unigram LM are truncated into 1000 sentences and 10 epochs respectively, and train the model without saving any file')
    parser.add_argument('--selectMode', default='normal', choices=['normal', 'sampling', 'top'],
                        help='when selecting sampling or top, the system truncate the vocabulary into the size of selectSize as introduced in Hiraoka2020. to make the small vocabulary, "Sampling" samples tokens from the vocabulary depending on the token probabilities, and "top" extract top-selectSize tokens from the distribution of tokens')
    parser.add_argument('--selectSize', default=8000, type=int,
                        help='the size of the restricted vocabulary introduced in Hiraoka2020')
    parser.add_argument('--useIndividualEmbed', action='store_true',
                        help='if true, the classifier and the neural unigram LM for tokenization use different word embeddings. if false, they share the same embedding')
    parser.add_argument('--freezeOTEmbedding', action='store_true',
                        help='freeze word embedding for optok\'s neural unigram LM for tokenization. embedding for CL is also freezed when the optok and the classifier share the embedding')
    parser.add_argument('--freezeCLEmbedding', action='store_true',
                        help='freeze word embedding for the classifier. embedding for optok is also freezed when the optok and the classifier share the embedding')
    parser.add_argument('--pretrainedEmbed', default=None, 
                        help='pretrained embedding path')
    parser.add_argument('--dropoutRate', default=0.5, type=float, help='dropout rate for the classifier. if bert, 0.1 is recommended')
    # inference parameters
    parser.add_argument('--nTest', default=1, type=int,
                        help='the value of nbest used in the inference phase. default is 1')
    parser.add_argument('--lamTest', default=1.0, type=float,
                        help='the value of lambda used in the inference phase. default is 1.0, meaning no effect')
    parser.add_argument('--selectModeTest', default='top', choices=['normal', 'sampling', 'top'],
                        help='when selecting sampling or top, the system creates the restricted vocabulary even in the inference. the optok of Hiraoka2020 restricted the vocabulary in the inference by extracting top-selectSize probabilities. "sampling" is not recommended due to the reproducibility problem.')
    parser.add_argument('--dumpEvalEachEpoch', action='store_true',
                        help='if true, dump the inference results for each epoch')
    args = parser.parse_args()

    # Copying the value of nbest to "mSample"
    # In the pilot study, we sample m tokenization from n best candidates
    # This is effective in some case, but the effect is not so large.
    # Therefore, we just set m to the same value of n, meaning no sampling.
    # We remain this value as a user's option but basically hide from the user.
    setattr(args, 'mSample', args.nbest)
    setattr(args, 'mTest', args.nTest)

    # construction
    if args.nbest < args.mSample:
        print('mSample size %d is larger than nbest size %d'%(args.mSample, args.nbest))
        print('replace mSample with nbest size')
        args.mSample = args.nbest

    timeStamp = util.getTimeStamp()
    datasetName = args.trainText.split('/')[-2]
    params4dir = {'maxEpoch': 'mxEp',
                  'batchSize': 'bs',
                  'uniLossWeight': 'ulw',
                  'lam': 'lam',
                  'nbest': 'nb', 
                  'mSample': 'ms',
                  'selectMode': 'slctM', 
                  'selectSize': 'slctS', 
                  'freezeEmbedding': 'fe'}

    vars_args = vars(args).copy()

    dirName = '%s/%s-%s'%(datasetName,
                          timeStamp,
                          '-'.join(['%s=%s'%(params4dir[str(k)],
                                             str(v).replace('/','+')) 
                                    for k,v in vars_args.items() if k in params4dir]))
    setattr(args, 'dirName', dirName)

    if not args.debug:
        # mkdir
        if not os.path.isdir('../results/'+datasetName):
            os.mkdir('../results/'+datasetName)
        os.mkdir('../results/'+dirName)
        # dump args
        yaml.dump(vars(args), open('../results/%s/args.yml'%dirName, 'w'))

    data, labels, mlm, cl, opt = prepare(args)
    cl = train(data, labels, mlm, cl, opt, args)

    if args.debug: exit()

    saveNLMasMLM(cl, mlm, '../results/%s/mlm.pickle'%dirName)
    
    cl.to('cpu')
    torch.save(cl.state_dict(), '../results/%s/cl.model'%dirName)
    
    dumpHistory('../results/%s/hist.txt'%dirName)
    

if __name__=='__main__':
    main()
