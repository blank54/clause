#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clauseutil import ClausePath, ClauseIO
clausepath = ClausePath()
clauseio = ClauseIO()

import json
import pandas as pd
import pickle as pk
from collections import defaultdict


def build_corpus(fname_data):
    fpath_data = os.path.join(clausepath.fdir_data, fname_data)

    corpus = []
    df = pd.read_excel(fpath_data)

    print('Build corpus')
    for _, row in df.iterrows():
        for idx, sent in enumerate(row['text'].split('  ')):
            tag = '{}_{:02d}'.format(row['tag'], idx+1)
            clause = Doc(tag=tag, text=sent)
            corpus.append(clause)

    print('\n  | Total {:,d} clauses'.format(len(corpus)))
    return corpus

def export_data_for_labeling(corpus):
    data = []
    for doc in corpus:
        data.append('  TAGSPLIT  '.join((doc.tag, doc.text)))

    fname_data_for_labeling = 'sent_for_labeling.txt'
    fpath_data_for_labeling = os.path.join(clausepath.fdir_data, fname_data_for_labeling)
    with open(fpath_data_for_labeling, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

    print('============================================================')
    print('Convert corpus to data_for_labeling')
    print('  | fdir : {}'.format(clausepath.fdir_data))
    print('  | fname: {}'.format(fname_data_for_labeling))

def read_labeled_data(fname_labeled_data):
    fpath_clause_labeled = os.path.join(clausepath.fdir_data, fname_labeled_data)
    labeled_data = []
    with open(fpath_clause_labeled, 'r') as f:
        for line in list(f):
            data = json.loads(line)
            if data['answer'] == 'accept':
                labeled_data.append(data)
            else:
                continue

    return labeled_data

def assign_labels(corpus, labeled_data):
    print('============================================================')
    print('Assign labels to corpus')
    print('  | # of corpus            : {:,}'.format(len(corpus)))
    print('  | # of labeled data      : {:,}'.format(len(labeled_data)))

    corpus_labeled = []
    for line in labeled_data:
        tag = line['text'].split('TAGSPLIT')[0].strip()
        doc = [doc for doc in corpus if doc.tag == tag][0]
        doc.labels = line['accept']
        corpus_labeled.append(doc)

    print('  | # of corpus with labels: {:,}'.format(len(corpus_labeled)))
    return corpus_labeled

def verify_labels(corpus):
    print('============================================================')
    print('Verify labels')

    for doc in corpus[:5]:
        print('--------------------------------------------------')
        print('  | Tag   : {}'.format(doc.tag))
        print('  | Text  : {}...'.format(doc.text[:50]))
        print('  | Labels: {}'.format(', '.join(doc.labels)))


if __name__ == '__main__':
    fname_data = 'clause.xlsx'
    fname_corpus = 'corpus.pk'
    fname_labeled_data = 'sent_labeled_2000.jsonl'

    ## Initialize corpus
    corpus = build_corpus(fname_data=fname_data)

    ## Export data for labeling
    export_data_for_labeling(corpus=corpus)

    ## Assign labels
    labeled_data = read_labeled_data(fname_labeled_data=fname_labeled_data)
    corpus_labeled = assign_labels(corpus=corpus, labeled_data=labeled_data)
    verify_labels(corpus=corpus_labeled)

    ## Save corpus
    print('============================================================')
    print('Save corpus')
    fname_corpus_labeled = 'corpus_{:,}.pk'.format(len(corpus_labeled))
    clauseio.save_corpus(corpus=corpus_labeled, fname_corpus=fname_corpus_labeled)