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

from copy import deepcopy
from collections import defaultdict


def export_corpus(corpus, LABEL_NAME, fname_corpus_sheet):
    corpus_sheet = defaultdict(list)
    for doc in corpus:
        corpus_sheet['tag'].append(doc.tag)
        corpus_sheet['text'].append(' '.join(doc.normalized_text))

        if LABEL_NAME in doc.labels:
            corpus_sheet['label'].append(1)
        else:
            corpus_sheet['label'].append(0)

    clauseio.save_corpus_sheet(corpus_sheet=corpus_sheet, fname=fname_corpus_sheet)


if __name__ == '__main__':
    print('============================================================')
    print('Prepare data for review')

    ## Filenames
    fname_corpus = 'corpus_preprocessed.pk'

    ## Corpus
    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)

    print(f'  | Corpus: {fname_corpus} ({len(corpus):,})')

    ## Run
    label_list = clauseio.read_label_list(version='v6')
    for LABEL_NAME in label_list:
        print('--------------------------------------------------')
        print(f'  | Label: {LABEL_NAME}')

        fname_corpus_sheet = f'corpus_for_review_{LABEL_NAME}.xlsx'

        ## Export current corpus
        export_corpus(corpus, LABEL_NAME, fname_corpus_sheet)