#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO
clauseio = ClauseIO()

import pickle as pk

from sklearn import preprocessing


if __name__ == '__main__':
    ## Parameters
    target_label = 'RnR'

    ## Filenames
    fname_corpus = 'corpus_940_T-t_P-t_N-t_S-t_L-t.pk'
    fname_d2v_model = 'd2v_940_V-100_W-5_M-5_E-100.pk'

    ## Load objects
    fpath_corpus = os.path.sep.join((clauseio.fdir_corpus, fname_corpus))
    with open(fpath_corpus, 'rb') as f:
        corpus = pk.load(f)

    fpath_d2v_model = os.path.sep.join((clauseio.fdir_model, fname_d2v_model))
    with open(fpath_d2v_model, 'rb') as f:
        d2v_model = pk.load(f)

    ## Data preparation
    X = [d2v_model.dv.get_vector(doc.tag) for doc in corpus]
    Y = [bool(target_label in doc.labels) for doc in corpus]

    # for doc, x, y in zip(corpus, X, Y):
    #     print(doc.tag)
    #     print(doc.labels)
    #     print(x[:5])
    #     print(y)
    #     print('--'*30)

    ## Data preprocess
    le = preprocessing.LabelEncoder()
    X_encoded = le.fit_transform(X)
    print(X_encoded)