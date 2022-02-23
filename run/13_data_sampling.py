#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClausePath, ClauseEval
clauseio = ClauseIO()
clausepath = ClausePath()
clauseeval = ClauseEval()

import random
from copy import deepcopy
from collections import defaultdict


def oversampling(corpus, LABEL_NAME):
    corpus_sampling = []
    for doc in corpus:
        corpus_sampling.append(doc)

        if LABEL_NAME in doc.labels:
            corpus_sampling.append(doc)
        else:
            continue

    clauseeval.label_distribution(corpus=corpus, show_only=LABEL_NAME)
    clauseeval.label_distribution(corpus=corpus_sampling, show_only=LABEL_NAME)

    return corpus_sampling

def downsampling(corpus, LABEL_NAME, sampling_ratio):
    corpus_sampling = []

    docs_pos, docs_neg = [], []
    for doc in corpus:
        if LABEL_NAME in doc.labels:
            docs_pos.append(doc)
        else:
            docs_neg.append(doc)

    cnt_pos = len(docs_pos)
    cnt_neg = len(docs_neg)
    cnt_pivot = min(int(min(cnt_pos, cnt_neg)*sampling_ratio), max(cnt_pos, cnt_neg))

    try:
        docs_pos_sampling = random.sample(docs_pos, cnt_pivot)
    except ValueError:
        docs_pos_sampling = docs_pos

    try:
        docs_neg_sampling = random.sample(docs_neg, cnt_pivot)
    except ValueError:
        docs_neg_sampling = docs_neg

    corpus_sampling = docs_pos_sampling+docs_neg_sampling

    clauseeval.label_distribution(corpus=corpus, show_only=LABEL_NAME)
    clauseeval.label_distribution(corpus=corpus_sampling, show_only=LABEL_NAME)


    return corpus_sampling

def reindexing(corpus):
    corpus_reindexed = []
    for idx, doc in enumerate(corpus):
        doc.tag = f'tag_{idx:04d}'
        corpus_reindexed.append(doc)

    return corpus_reindexed


if __name__ == '__main__':
    print('============================================================')
    print('Sampling')

    ## Filenames
    fname_corpus = 'corpus_revised.pk'
    fname_corpus_sampled = 'corpus_sampled.pk'

    ## Corpus
    print('--------------------------------------------------')
    print('Corpus label distribution (BEFORE)')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    label_cnt = clauseeval.label_distribution(corpus=corpus)

    ## Parameters
    print('--------------------------------------------------')
    print('Sampling parameters')

    OVERSAMPLING_LIMIT = 0.1
    DOWNSAMPLING_RATIO = 1.5

    min_cnt = int(0.1 * len(corpus))
    labels_for_oversampling = [LABEL_NAME for LABEL_NAME, cnt in label_cnt.items() if cnt < min_cnt]
    
    print(f'  | Oversampling: {OVERSAMPLING_LIMIT} --> {min_cnt}')

    ## Sampling
    corpus_dict = {}

    label_list = clauseio.read_label_list(version='v6')
    for LABEL_NAME in label_list:
        print('--------------------------------------------------')

        if LABEL_NAME in labels_for_oversampling:
            print('OVERSAMPLING')
            corpus_over = deepcopy(oversampling(corpus=corpus, LABEL_NAME=LABEL_NAME))
        else:
            corpus_over = deepcopy(corpus)

        print('DOWNSAMPLING')
        corpus_down = deepcopy(downsampling(corpus=corpus_over, LABEL_NAME=LABEL_NAME, sampling_ratio=DOWNSAMPLING_RATIO))

        corpus_dict[LABEL_NAME] = reindexing(corpus=corpus_down)

    ## Export sampled corpus
    print('============================================================')
    print('Export corpus_dict')

    clauseio.save_corpus(corpus_dict, fname_corpus_sampled)