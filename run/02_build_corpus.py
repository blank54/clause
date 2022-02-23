#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clauseutil import ClausePath, ClauseIO, ClauseEval
clausepath = ClausePath()
clauseio = ClauseIO()
clauseeval = ClauseEval()

import json
from collections import defaultdict


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
    corpus_labeled = []
    for line in labeled_data:
        tag = line['text'].split('TAGSPLIT')[0].strip()
        doc = [doc for doc in corpus if doc.tag == tag][0]
        doc.labels = line['accept']

        if any((('METHOD' in doc.labels), ('CONDITIN' in doc.labels), ('PROCEDURE' in doc.labels))):
            doc.labels.append('ACTION')
        else:
            pass

        corpus_labeled.append(doc)

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
    print('============================================================')
    print('Assign label to corpus')

    ## Filenames
    fname_labeled_data = 'clause_labeled.jsonl'
    fname_corpus = 'corpus.pk'
    fname_corpus_labeled = 'corpus_labeled.pk'

    ## Load corpus
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)

    print('  | # of docs: {:,}'.format(len(corpus)))

    ## Load data labeled
    print('--------------------------------------------------')
    print('Load labeled data')

    labeled_data = read_labeled_data(fname_labeled_data=fname_labeled_data)

    print('  | # of labeled data: {:,}'.format(len(labeled_data)))

    ## Assign labels
    print('--------------------------------------------------')
    print('Assign labels')

    corpus_labeled = assign_labels(corpus=corpus, labeled_data=labeled_data)
    # verify_labels(corpus=corpus_labeled)

    print('  | # of labeled docs: {:,}'.format(len(corpus_labeled)))

    ## Save corpus
    print('--------------------------------------------------')
    print('Save labeled corpus')

    clauseio.save_corpus(corpus=corpus_labeled, fname_corpus=fname_corpus_labeled)

    ## Label distribution
    print('--------------------------------------------------')
    print('Label distribution')

    clauseeval.label_distribution(corpus=corpus_labeled)