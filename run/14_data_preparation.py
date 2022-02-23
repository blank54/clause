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


def make_directories(LABEL_NAME):
    for data_type in ['train', 'test']:
        for class_type in ['pos', 'neg']:
            os.makedirs(os.path.sep.join((clausepath.fdir_data, LABEL_NAME, data_type, class_type)), exist_ok=True)

def save_doc_here(fdir, fname, doc):
    fpath = os.path.sep.join((fdir, fname))
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(' '.join(doc.normalized_text))

def build_dataset(corpus, option, LABEL_NAME):
    for doc in corpus:
        fname = f'{doc.tag}.txt'
        if LABEL_NAME in doc.labels:
            fdir = os.path.sep.join((clausepath.fdir_data, LABEL_NAME, option, 'pos'))
        else:
            fdir = os.path.sep.join((clausepath.fdir_data, LABEL_NAME, option, 'neg'))

        save_doc_here(fdir, fname, doc)


if __name__ == '__main__':
    print('============================================================')
    print('Build datasets')

    ## Filenames
    fname_corpus = 'corpus_sampled.pk'

    ## Corpus
    corpus_dict = clauseio.read_corpus(fname_corpus=fname_corpus)

    ## Parameters
    RANDOM_STATE = 42
    TRAIN_TEST_RATIO = 0.8

    ## Run for each label
    for LABEL_NAME, corpus in corpus_dict.items():
        print(f'  | Dataset for {LABEL_NAME} ... ', end='')

        corpus_train, corpus_test = train_test_split(corpus, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)

        ## Directory
        make_directories(LABEL_NAME=LABEL_NAME)

        ## Dataset
        build_dataset(corpus=corpus_train, option='train', LABEL_NAME=LABEL_NAME)
        build_dataset(corpus=corpus_test, option='test', LABEL_NAME=LABEL_NAME)

        print('done!')