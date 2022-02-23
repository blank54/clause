#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClausePath
clauseio = ClauseIO()
clausepath = ClausePath()

from sklearn.model_selection import train_test_split


def make_directories(label_version):
    label_list = clauseio.read_label_list(version=label_version)
    for label_name in label_list:
        os.makedirs(os.path.sep.join((clausepath.fdir_data, 'default', label_name, 'train', 'pos')), exist_ok=True)
        os.makedirs(os.path.sep.join((clausepath.fdir_data, 'default', label_name, 'train', 'neg')), exist_ok=True)
        os.makedirs(os.path.sep.join((clausepath.fdir_data, 'default', label_name, 'test', 'pos')), exist_ok=True)
        os.makedirs(os.path.sep.join((clausepath.fdir_data, 'default', label_name, 'test', 'neg')), exist_ok=True)

def save_doc_here(fdir, fname, doc):
    fpath = os.path.sep.join((fdir, fname))
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(' '.join(doc.normalized_text))

def build_dataset(corpus, option, label_version):
    label_list = clauseio.read_label_list(version=label_version)
    for doc in corpus:
        fname = f'{doc.tag}.txt'
        for label_name in label_list:
            if label_name in doc.labels:
                fdir = os.path.sep.join((clausepath.fdir_data, 'default', label_name, option, 'pos'))
            else:
                fdir = os.path.sep.join((clausepath.fdir_data, 'default', label_name, option, 'neg'))

            if any((('METHOD' in doc.labels), ('CONDITION' in doc.labels), ('PROCEDURE' in doc.labels))):
                fdir_action = os.path.sep.join((clausepath.fdir_data, 'default', 'ACTION', option, 'pos'))
            else:
                fdir_action = os.path.sep.join((clausepath.fdir_data, 'default', 'ACTION', option, 'neg'))

            save_doc_here(fdir, fname, doc)
            save_doc_here(fdir_action, fname, doc)


if __name__ == '__main__':
    ## Filenames
    base = '1,976'
    fname_corpus = 'corpus_{}_T-t_P-t_N-t_S-t_L-t.pk'.format(base)

    ## Parameters
    RANDOM_STATE = 42
    TRAIN_TEST_RATIO = 0.8

    label_version_small = 'v2'
    label_version_large = 'v6'

    ## Directory
    make_directories(label_version=label_version_large)

    ## Dataset
    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    corpus_train, corpus_test = train_test_split(corpus, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)

    build_dataset(corpus=corpus_train, option='train', label_version=label_version_large)
    build_dataset(corpus=corpus_test, option='test', label_version=label_version_large)