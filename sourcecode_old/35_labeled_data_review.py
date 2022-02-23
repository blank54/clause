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


def export_corpus(corpus, LABEL_NAME, fname_corpus_sheet_before):
    corpus_sheet = defaultdict(list)
    for doc in corpus:
        corpus_sheet['tag'].append(doc.tag)
        corpus_sheet['text'].append(' '.join(doc.normalized_text))

        if LABEL_NAME in doc.labels:
            corpus_sheet['label'].append(1)
        else:
            corpus_sheet['label'].append(0)

    clauseio.save_corpus_sheet(corpus_sheet=corpus_sheet, fname=fname_corpus_sheet_before)

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
    ## Parameters
    LABEL_NAME = 'PAYMENT'
    DO_REVISE = False

    ## Filenames
    base = '1,976'

    fname_corpus_before = f'corpus_{base}_T-t_P-t_N-t_S-t_L-t.pk'
    fname_corpus_sheet_before = f'corpus_{base}_{LABEL_NAME}.xlsx'

    fname_corpus_sheet_after = f'corpus_{base}_{LABEL_NAME}_revised.xlsx'
    fname_corpus_after = f'corpus_{base}_T-t_P-t_N-t_S-t_L-t_revised.pk'

    ## Import current corpus
    print('============================================================')
    print('Import current corpus')
    print(f'  | Fname: {fname_corpus_before}')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus_before)

    print(f'  | Len  : {len(corpus):,}')

    if DO_REVISE:
        ## Export current corpus
        print('============================================================')
        print('Export current corpus in .xslx')
        print(f'  | Fname: {fname_corpus_sheet_before}')

        export_corpus(corpus, LABEL_NAME, fname_corpus_sheet_before)

    else:
        ## Import revised corpus sheet
        print('============================================================')
        print('Import revised corpus sheet')
        print(f'  | Fname: {fname_corpus_sheet_before}')

        corpus_sheet_revised = clauseio.read_corpus_sheet(fname=fname_corpus_sheet_after)
        corpus_revised, revisions = revise_corpus(corpus, corpus_sheet_revised, LABEL_NAME)
        print(revisions)

        show_revisions(corpus, corpus_revised, revisions)

        ## Export revised corpus
        print('============================================================')
        print('Export current corpus in .pk')

        clauseio.save_corpus(corpus_revised, fname_corpus_after)