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

import pandas as pd
from copy import deepcopy
from collections import defaultdict


def revise_corpus(corpus, corpus_sheet_revised, LABEL_NAME):
    corpus_revised = []
    revisions = []

    for doc in corpus:
        doc2 = deepcopy(doc)
        if doc2.tag not in list(corpus_sheet_revised['tag']):
            corpus_revised.append(doc2)

        else:
            revised_label = int(corpus_sheet_revised.loc[corpus_sheet_revised['tag']==doc2.tag,'label'])
            if LABEL_NAME in doc2.labels:
                if revised_label == 1:
                    pass
                else:
                    doc2.labels.pop(LABEL_NAME)
                    revisions.append(doc2.tag)
            else:
                if revised_label == 1:
                    doc2.labels.append(LABEL_NAME)   
                    revisions.append(doc2.tag)
                else:
                    pass

            doc2.labels = list(set(doc2.labels))
            corpus_revised.append(doc2)

    return corpus_revised, revisions

def show_revisions(corpus, corpus_revised, revisions):
    for doc_before, doc_after in zip(corpus, corpus_revised):
        if doc_before.tag not in revisions:
            continue
        else:
            print('--------------------------------------------------')
            print(f'  | Tag            : {doc_after.tag}')
            print('  | Labels (before): {}'.format(', '.join(list(sorted(doc_before.labels)))))
            print('  | Labels (after) : {}'.format(', '.join(list(sorted(doc_after.labels)))))


if __name__ == '__main__':
    print('============================================================')
    print('Revise corpus')

    ## Filenames
    fname_corpus = 'corpus_preprocessed.pk'
    fname_corpus_revised = 'corpus_revised.pk'

    ## Corpus
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)

    print('  | # of docs: {:,}'.format(len(corpus)))

    ## Run
    label_list = clauseio.read_label_list(version='v6')
    for LABEL_NAME in label_list:
        print('--------------------------------------------------')
        print(f'  | Label: {LABEL_NAME}')

        try:
            fname_corpus_sheet = f'corpus_for_review_{LABEL_NAME}_revised.xlsx'
            corpus_sheet_revised = clauseio.read_corpus_sheet(fname=fname_corpus_sheet)
            corpus, revisions = deepcopy(revise_corpus(corpus, corpus_sheet_revised, LABEL_NAME))

            # show_revisions(corpus, corpus_revised, revisions)
            print(f'  | revised docs: {len(revisions):,}')
        except FileNotFoundError:
            print('  | NO REVISED CORPUS...')

    ## Export revised corpus
    print('============================================================')
    print('Export current corpus in .pk')

    clauseio.save_corpus(corpus, fname_corpus_revised)