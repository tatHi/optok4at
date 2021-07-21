
# Requirements
- [multigram v0.2.1](https://github.com/tatHi/multigram/releases/tag/v0.2.1)
- numpy==1.20.3
- PyYAML==5.3
- sentencepiece==0.1.91
- torch==1.6.0+cu101
- tqdm==4.43.0

# Preparation
## Installation 
Pip-install the modified fairseq and optok_nmt into your environment.
If your environment already has fairseq, please `pip uninstall fairseq` at the first.

```
$ cd optok
$ pip install --editable .
$ cd ../fairseq
$ pip install --editable .
```

## Prepare datasets
- Prepare raw-texts for machine translation
- Create SentencePiece models with the training split referring to [the official repository](https://github.com/google/sentencepiece)
  - SentncePiece models are used to tokenize the text and initialization of OpTok.
- Tokenize the raw-text and preprocess it with `fairseq-preprocess` referring to [the fairseq tutorial](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model)

## Training
Because the training of an NMT model bases on the default training of fairseq, I reccomend you to read [the fairseq tutorial](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) if you are not familiar with fairseq.
The differences from the vanilla fairseq are:
- Use `transformer_optok` (developed in this work) as architecture
- Use `optok_pass_through` (developed in this work) as criterion
- Specify SentencePiece models through arguments

The command for the training of fairseq with Optok is:

```
$ DATA=/path/to/data
$ SPSRC=/path/to/sentencepiece_model_for_source
$ SPTGT=/path/to/sentencepiece_model_for_target
$ CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $DATA \
    --seed 1234 \
    --clip-norm 0.0 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --max-tokens 1000 \
    --max-epoch 150 \
    --update-freq 1 \
    --arch transformer_optok \
    --save-dir checkpoints \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --criterion optok_pass_through \
    --optok-sp-src-path $SPSRC\
    --optok-sp-tgt-path $SPTGT \
    --optok-m 3 \
    --ddp-backend=no_c10d \
    --sp-alpha 0.2 \
    --keep-last-epochs 10
```

The above training saves cache files of the pre-trained neural unigram language model and word embeddings for tokenization in `/checkpoints/cache/`.
As introduced in Hiraoka2020, we pre-train the neural unigram language model using KL-Loss against the original word probabilities of the SentencePiece model.
This pre-training takes a bit long time, so using the cached model for the next time if you conduct same experiments on the same dataset as the following options:

```
--optok-lmemb-enc /checkpoints/cache/source.model.lmemb.pt 
--optok-nlm-enc /checkpoints/cache/source.model.nlm.pt
--optok-lmemb-dec /checkpoints/cache/target.model.lmemb.pt
--optok-nlm-dec /checkpoints/cache/target.model.nlm.pt
```

The training process outputs the following two language models for the tokenizer in addition to the original translation model.
- checkpoint_best.optok.enc.mlm
- checkpoint_best.optok.dec.mlm

You can tokenize sentences with mlm files using the multigram package:
```
$ python
>>> from multigram import lm
>>> from multigram import tokenizer
>>> mlm = lm.MultigramLM()
>>> mlm.load('path/to/checkpoint_best.optok.enc.mlm')
>>> tknzr = tokenizer.Tokenizer(mlm)
>>> tknzr.encode_as_pieces('i love you')
['i', ' ', 'love', ' ', 'y', 'o', 'u']
```

## Inference
The way of the inference using the trained model is completely same as the original fairseq process.
```
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
        $DATA \
        --path /path/to/checkpoint_best.pt \
        --batch-size 128 \
        --beam 5 \
        --remove-bpe sentencepiece > /path/to/best_output.log
```

# Options
Additional arguments for OpTok are following (you can check them in `/fairseq/fairseq/models/transformer_optok.py`):

```
--optok-sp-src-path     path to sentnecepiece model to train optok lm's theta for source language
--optok-sp-tgt-path     path to sentnecepiece model to train optok lm\'s theta for target language')
--sp-alpha              alpha for sentencepiece\'s subword regularization')
--optok-m               the number of N-best tokenization condidates corresponding to N in our paper')
--optok-enc-mu          the value to weight a loss to maintain the unigram language model for encoder')
--optok-dec-mu          the value to weight a loss to maintain the unigram language model for decoder')
--optok-fitto           if true, the neural unigram language model is pre-trained at the beginning of the training')
--optok-normal-enc      if true, the system uses sentencepiece+subword regularization instead of optok for the encoder')
--optok-normal-dec      if true, the system uses sentencepiece+subword regularization instead of optok for the decoder')
--optok-lmemb-enc       pre-trained embeddings for encoder-side language model
--optok-lmemb-dec       pre-trained embeddings for decoder-side language model
--optok-nlm-enc         pre-trained parameters of ecndoer-side language model
--optok-nlm-dec         pre-trained parameters of decoder-side langauge model
```
