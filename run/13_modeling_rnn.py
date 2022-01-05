#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
https://justkode.kr/deep-learning/pytorch-rnn
https://ichi.pro/ko/pytorchleul-sayonghan-lstm-tegseuteu-bunlyu-48844070948800
'''

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClauseFunc, ClauseEval
clauseio = ClauseIO()
clausefunc = ClauseFunc()
clauseeval = ClauseEval()

import pickle as pk

import torch
import torch.nn as nn








if __name__ == '__main__':
    ## Filenames
    base = '1,053'
    fname_d2v = 'd2v_{}_V-100_W-5_M-5_E-100.pk'.format(base)
    fname_corpus = 'corpus_{}_T-t_P-t_N-t_S-t_L-t.pk'.format(base)

    ## Parameters
    RESAMPLING = False
    MAX_SENT_LEN = 128
    BATCH_SIZE = 8
    RANDOM_STATE = 42

    EPOCHS = 100
    LEARNING_RATE = 2e-4

    ## Embedding
    with open(os.path.sep.join((clauseio.fdir_model, fname_d2v)), 'rb') as f:
        d2v_model = pk.load(f)

    ## Data preparation
    target_label = 'TEMPORAL'
    fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
    fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))
    with open(fpath_resampled, 'rb') as f:
        corpus_res = pk.load(f)

    train_inputs, train_masks, train_labels = corpus_res['train']
    train_dataloader = build_dataloader(inputs=train_inputs, labels=train_labels, masks=train_masks, target_label=target_label, encode=False)

    print(type(train_inputs))
    print(len(train_inputs))

    ## Model Identification
