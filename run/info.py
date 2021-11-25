#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clauseutil import ClausePath, ClauseIO, ClauseFunc
clausepath = ClausePath()
clauseio = ClauseIO()
clausefunc = ClauseFunc()

import itertools
import pickle as pk
from collections import defaultdict


if __name__ == '__main__':
    fname_corpus = 'corpus.pk'

    print('============================================================')
    print('Corpus')

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    
    print('  | # of clauses        : {:,}'.format(len(corpus)))
    print('  | # of sentences         : {:,}'.format(len(list(itertools.chain(*[p.text.split('  ') for p in corpus])))))

    print('============================================================')
    print('Data exploration')

    for target_label in ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'RnR', 'DEFINITION', 'SCOPE']:
        labels_encoded = clausefunc.encode_labels_binary(labels=[p.labels for p in corpus], target_label=target_label)

        positive = int(sum(labels_encoded))
        negative = int(len(labels_encoded)) - positive

        print('  | {:10}: Positive-{:3,} & Negative-{:3,}'.format(target_label, positive, negative))