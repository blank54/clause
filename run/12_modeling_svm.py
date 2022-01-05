#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
http://www.gisdeveloper.co.kr/?p=8180
https://stackoverflow.com/questions/19629331/python-how-to-find-accuracy-result-in-svm-text-classifier-algorithm-for-multil
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_import(base):
    fname_corpus = 'corpus_{}_T-t_P-t_N-t_S-t_L-t.pk'.format(base)
    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)

    fname_d2v = 'd2v_{}_V-100_W-5_M-5_E-100.pk'.format(base)
    with open(os.path.sep.join((clauseio.fdir_model, fname_d2v)), 'rb') as f:
        d2v_model = pk.load(f)

    return corpus, d2v_model

def data_preparation(corpus, d2v_model):
    vectors, labels = [], []
    for doc in corpus:
        vectors.append(d2v_model.dv[doc.tag])
        labels.append(doc.labels)

    return vectors, labels


if __name__ == '__main__':
    ## Filenames
    base = '1,053'
    
    ## Parameters
    TRAIN_VALID_RATIO = 0.7
    RANDOM_STATE = 42

    ## Data import
    corpus, d2v_model = data_import(base=base)

    ## Model development
    print('============================================================')
    print('SVM model accuracy')

    vectors, labels = data_preparation(corpus=corpus, d2v_model=d2v_model)
    label_list = ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'RnR', 'DEFINITION', 'SCOPE']
    for target_label in label_list:
        target_labels = clausefunc.encode_labels_binary(labels=labels, target_label=target_label)
        train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(vectors, target_labels, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

        classifier = svm.SVC(kernel='linear')
        classifier.fit(train_inputs, train_labels)

        predicted = classifier.predict(valid_inputs)
        accuracy = accuracy_score(valid_labels, predicted)

        print('  | {:10}: {:.03f}'.format(target_label, accuracy))