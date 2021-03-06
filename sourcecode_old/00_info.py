#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clauseutil import ClausePath, ClauseIO, ClauseFunc
clausepath = ClausePath()
clauseio = ClauseIO()
clausefunc = ClauseFunc()

import itertools
import numpy as np
import pickle as pk
from copy import deepcopy
from collections import defaultdict


def get_accuracy_by_loss(result):
    final_loss = 999
    for idx, row in result.iterrows():
        current_loss = row['train_loss'] + row['valid_loss']

        if current_loss <= final_loss:
            final_loss = deepcopy(current_loss)
            epoch = row['epoch']
            valid_accuracy = row['valid_accuracy']
            test_accuracy = row['test_accuracy']
        else:
            continue

    return epoch, valid_accuracy, test_accuracy

def get_accuracy_by_acc(result):
    final_acc = 0
    for idx, row in result.iterrows():
        current_acc = (row['valid_accuracy'] + row['test_accuracy'])/2

        if current_acc >= final_acc:
            epoch = row['epoch']
            valid_accuracy = row['valid_accuracy']
            test_accuracy = row['test_accuracy']
        else:
            continue

    return epoch, valid_accuracy, test_accuracy


if __name__ == '__main__':
    ## Filenames
    base = '1,976'
    label_list_version = 'v2'
    fname_corpus = 'corpus_{}.pk'.format(base)
    fname_result_base = 'result_{}_TR-60_VL-20_TS-20_BS-16_EP-1000_LR-0.0002_LB'.format(base)
    label_list = clauseio.read_label_list(version=label_list_version)

    ## Status information
    print('============================================================')
    print('Corpus')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    
    print('  | # of clauses  : {:,}'.format(len(corpus)))
    print('  | # of sentences: {:,}'.format(len(list(itertools.chain(*[p.text.split('  ') for p in corpus])))))


    print('============================================================')
    print('Data exploration')
    print('  | label       positive  negative')

    
    for target_label in label_list:
        labels_encoded = clausefunc.encode_labels_binary(labels=[p.labels for p in corpus], target_label=target_label)

        positive = int(sum(labels_encoded))
        negative = int(len(labels_encoded)) - positive

        print('  | {:10}:    {:5,}     {:5,}'.format(target_label, positive, negative))


    print('============================================================')
    print('Data resampling on training dataset')
    print('  |             Before resampling           After resampling')
    print('  | label       positive  negative          positive  negative')

    for target_label in label_list:
        fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
        fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))
        with open(fpath_resampled, 'rb') as f:
            corpus_res = pk.load(f)

        train_inputs, train_masks, train_labels = corpus_res['train']
        train_inputs_res, train_masks_res, train_labels_res = corpus_res['train_res']
        valid_inputs, valid_masks, valid_labels = corpus_res['valid']
        valid_inputs_res, valid_masks_res, valid_labels_res = corpus_res['valid_res']
        test_inputs, test_masks, test_labels = corpus_res['test']
        test_inputs_res, test_masks_res, test_labels_res = corpus_res['test_res']

        full_labels = train_labels+valid_labels+test_labels
        full_labels_res = train_labels_res+valid_labels_res+test_labels_res
        
        total = len(full_labels)
        pos = sum(clausefunc.encode_labels_binary(full_labels, target_label=target_label))
        neg = total - pos

        total_res = len(full_labels_res)
        pos_res = sum(full_labels_res)
        neg_res = total_res - pos_res

        print('  | {:10}:    {:5,}     {:5,} ({:,}) ->    {:5,}     {:5,} ({:,})'.format(target_label, pos, neg, total, pos_res, neg_res, total_res))


    print('============================================================')
    print('BERT result summary')
    print('  | label      epoch  valid_acc  test_acc')

    valid_accuracy_list = []
    test_accuracy_list = []    
    for target_label in label_list:
        fname_result = '{}-{}.xlsx'.format(fname_result_base, target_label)
        result = clauseio.read_result(fname_result=fname_result)
        epoch, valid_accuracy, test_accuracy = get_accuracy_by_acc(result=result)
        valid_accuracy_list.append(valid_accuracy)
        test_accuracy_list.append(test_accuracy)

        print('  | {:10}  {:4}      {:.03f}     {:.03f}'.format(target_label, epoch, valid_accuracy, test_accuracy))

    print('  | AVERAGE               {:.03f}     {:.03f}'.format(np.mean(valid_accuracy_list), np.mean(test_accuracy_list)))