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

import itertools
import pickle as pk
from collections import defaultdict


if __name__ == '__main__':
    fname_corpus = 'provision.pk'
    fname_corpus_labeled = 'provision_labeled.pk'

    print('============================================================')
    print('Corpus')

    corpus = provfunc.read_corpus(fname_corpus=fname_corpus)
    corpus_labeled = provfunc.read_corpus(fname_corpus=fname_corpus_labeled)
    
    print('  | # of provisions        : {:,}'.format(len(corpus)))
    print('  | # of provisions labeled: {:,}'.format(len(corpus_labeled)))
    print('  | # of sentences         : {:,}'.format(len(list(itertools.chain(*[p.text.split('  ') for p in corpus])))))

    print('============================================================')
    print('Data exploration')

    for target_label in ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'RnR', 'DEFINITION', 'SCOPE']:
        labels_encoded = provfunc.encode_labels_binary(labels=[p.labels for p in corpus_labeled], target_label=target_label)

        positive = int(sum(labels_encoded))
        negative = int(len(labels_encoded)) - positive

        print('  | {:10}: Positive-{:3,} & Negative-{:3,}'.format(target_label, positive, negative))