# Requirements
- [multigram v0.2.1](https://github.com/tatHi/multigram/releases/tag/v0.2.1)
- numpy==1.20.3
- PyYAML==5.3
- scikit-learn==0.22.2
- scipy==1.5.2
- sentencepiece==0.1.91
- torch==1.6.0+cu101
- tqdm==4.43.0

# Quick Start
You can try the system with a sample dataset as following:

```
$ mkdir results
$ cd src
$ CUDA_VISIBLE_DEVICES=-1 python train.py \
    --trainText ../sample-data/train.text \
    --trainLabel ../sample-data/train.label \
    --validText ../sample-data/valid.text \
    --validLabel ../sample-data/valid.label \
    --evalText ../sample-data/test.text \
    --evalLabel ../sample-data/test.label \
    --maxEpoch 2 \
    --batchSize 4 \
    --nbest 3
```

This trial dumps the following files under `/results/sample-data/DATE-ID-OPTION/`:
- args.yml
    - options of the training  
- bestpreds.pickle  
    - predictions of the best model for the test split
- cl.best.model  
    - parameters of the best classifier depending on the validation split
- cl.model  
    - parameters of the classifier after the max-epoch training
- hist.txt
    - log of total loss and F1 scores on valid/test splits
- mlm.best.pickle  
    - parameters of multigram lm of the best model
- mlm.pickle
    - parameters of multigram lm after the max-epoch training

You can tokenize sentences with `mlm.pickle` using the multigram package:
```
$ python
>>> from multigram import lm
>>> from multigram import tokenizer
>>> mlm = lm.MultigramLM()
>>> mlm.load('path/to/mlm.pickle')
>>> tknzr = tokenizer.Tokenizer(mlm)
>>> tknzr.encode_as_pieces('i love you')
['i', ' ', 'love', ' ', 'y', 'o', 'u']
```

# Options
```
$ python train.py -h
usage: train.py [-h] [-tt TRAINTEXT] [-tl TRAINLABEL] [-vt VALIDTEXT]
                [-vl VALIDLABEL] [-et EVALTEXT] [-el EVALLABEL]
                [-mlm MULTIGRAMLANGUAGEMODEL] [-spm SENTENCEPIECEMODEL]
                [-me MAXEPOCH] [-bs BATCHSIZE] [-ulw UNILOSSWEIGHT] [-l LAM]
                [-nb NBEST] [--debug] [--selectMode {normal,sampling,top}]
                [--selectSize SELECTSIZE] [--useIndividualEmbed]
                [--freezeOTEmbedding] [--freezeCLEmbedding]
                [--pretrainedEmbed PRETRAINEDEMBED]
                [--dropoutRate DROPOUTRATE] [--nTest NTEST]
                [--lamTest LAMTEST] [--selectModeTest {normal,sampling,top}]
                [--dumpEvalEachEpoch]

optional arguments:
  -h, --help            show this help message and exit
  -tt TRAINTEXT, --trainText TRAINTEXT
  -tl TRAINLABEL, --trainLabel TRAINLABEL
  -vt VALIDTEXT, --validText VALIDTEXT
  -vl VALIDLABEL, --validLabel VALIDLABEL
  -et EVALTEXT, --evalText EVALTEXT
  -el EVALLABEL, --evalLabel EVALLABEL
  -mlm MULTIGRAMLANGUAGEMODEL, --multigramLanguageModel MULTIGRAMLANGUAGEMODEL
                        multigram language model for tokenization
  -spm SENTENCEPIECEMODEL, --sentencePieceModel SENTENCEPIECEMODEL
                        sentencepiece model for tokenization
  -me MAXEPOCH, --maxEpoch MAXEPOCH
  -bs BATCHSIZE, --batchSize BATCHSIZE
  -ulw UNILOSSWEIGHT, --uniLossWeight UNILOSSWEIGHT
                        weight for loss value to maintain unigram language
                        model introduced in Hiraoka2020
  -l LAM, --lam LAM     smoothing coefficient for a distribution to sample
                        restricted vocabulary introduced in Hiraoka2020
  -nb NBEST, --nbest NBEST
  --debug               if true, the size of corpus and the pre-training epoch
                        of the initized neural unigram LM are truncated into
                        1000 sentences and 10 epochs respectively, and train
                        the model without saving any file
  --selectMode {normal,sampling,top}
                        when selecting sampling or top, the system truncate
                        the vocabulary into the size of selectSize as
                        introduced in Hiraoka2020. to make the small
                        vocabulary, "Sampling" samples tokens from the
                        vocabulary depending on the token probabilities, and
                        "top" extract top-selectSize tokens from the
                        distribution of tokens
  --selectSize SELECTSIZE
                        the size of the restricted vocabulary introduced in
                        Hiraoka2020
  --useIndividualEmbed  if true, the classifier and the neural unigram LM for
                        tokenization use different word embeddings. if false,
                        they share the same embedding
  --freezeOTEmbedding   freeze word embedding for optok's neural unigram LM
                        for tokenization. embedding for CL is also freezed
                        when the optok and the classifier share the embedding
  --freezeCLEmbedding   freeze word embedding for the classifier. embedding
                        for optok is also freezed when the optok and the
                        classifier share the embedding
  --pretrainedEmbed PRETRAINEDEMBED
                        pretrained embedding path
  --dropoutRate DROPOUTRATE
                        dropout rate for the classifier. if bert, 0.1 is
                        recommended
  --nTest NTEST         the value of nbest used in the inference phase.
                        default is 1
  --lamTest LAMTEST     the value of lambda used in the inference phase.
                        default is 1.0, meaning no effect
  --selectModeTest {normal,sampling,top}
                        when selecting sampling or top, the system creates the
                        restricted vocabulary even in the inference. the optok
                        of Hiraoka2020 restricted the vocabulary in the
                        inference by extracting top-selectSize probabilities.
                        "sampling" is not recommended due to the
                        reproducibility problem.
  --dumpEvalEachEpoch   if true, dump the inference results for each epoch
```
