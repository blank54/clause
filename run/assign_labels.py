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
import pickle as pk


def read_labeled_data(fname_provision_labeled):
    fpath_provision_labeled = os.path.join(provpath.fdir_data, fname_provision_labeled)
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

def verify_labels(corpus_labeled):
    print('============================================================')
    print('Verify labels')

    for doc in corpus_labeled[:5]:
        print('--------------------------------------------------')
        print('  | Tag   : {}'.format(doc.tag))
        print('  | Text  : {}...'.format(doc.text[:50]))
        print('  | Labels: {}'.format(', '.join(doc.labels)))


if __name__ == '__main__':
    fname_corpus = 'provision.pk'
    fname_corpus_labeled = 'provision_labeled.pk'
    fname_provision_labeled = 'provision_labeled.jsonl'

    corpus = provfunc.read_corpus(fname_corpus=fname_corpus)
    labeled_data = read_labeled_data(fname_provision_labeled=fname_provision_labeled)

    corpus_labeled = assign_labels(corpus=corpus, labeled_data=labeled_data)
    provfunc.save_corpus(corpus=corpus_labeled, fname_corpus=fname_corpus_labeled)

    verify_labels(corpus_labeled)