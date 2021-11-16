#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from provutil import ProvPath, ProvFunc
provpath = ProvPath()
provfunc = ProvFunc()

import json
import pandas as pd
import pickle as pk
from collections import defaultdict


def build_corpus(fname_data):
    fpath_data = os.path.join(provpath.fdir_data, fname_data)

    corpus = []
    df = pd.read_excel(fpath_data)

    print('Build corpus')
    for idx, row in df.iterrows():
        provision = Doc(tag=row['tag'], text=row['text'])
        corpus.append(provision)

    print('\n  | Total {:,d} provisions'.format(len(corpus)))
    return corpus

def export_data_for_labeling(corpus):
    data = []
    for doc in corpus:
        data.append('  TAGSPLIT  '.join((doc.tag, doc.text)))

    fname_data_for_labeling = 'provision_for_labeling.txt'
    fpath_data_for_labeling = os.path.join(provpath.fdir_data, fname_data_for_labeling)
    with open(fpath_data_for_labeling, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

    print('============================================================')
    print('Convert corpus to data_for_labeling')
    print('  | fdir : {}'.format(provpath.fdir_data))
    print('  | fname: {}'.format(fname_data_for_labeling))

def read_labeled_data(fname_labeled_data):
    fpath_provision_labeled = os.path.join(provpath.fdir_data, fname_labeled_data)
    with open(fpath_provision_labeled, 'r') as f:
        labeled_data = [json.loads(line) for line in list(f)]
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
    fname_data = 'provision.xlsx'
    fname_corpus = 'corpus.pk'
    fname_labeled_data = 'provision_labeled.jsonl'

    ## Initialize corpus
    corpus = build_corpus(fname_data=fname_data)

    ## Export data for labeling
    export_data_for_labeling(corpus=corpus)

    ## Assign labels
    labeled_data = read_labeled_data(fname_labeled_data=fname_labeled_data)
    corpus_labeled = assign_labels(corpus=corpus, labeled_data=labeled_data)
    provfunc.save_corpus(corpus=corpus_labeled, fname_corpus=fname_corpus)
    verify_labels(corpus=corpus_labeled)