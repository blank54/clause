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


def read_corpus(fname_corpus):
    fpath_corpus = os.path.join(provpath.fdir_corpus, fname_corpus)
    with open(fpath_corpus, 'rb') as f:
        corpus = pk.load(f)
    return corpus


if __name__ == '__main__':
    fname_corpus = 'provision.pk'
    corpus = read_corpus(fname_corpus=fname_corpus)

    print('============================================================')
    print('Corpus')
    print('  | # of provisions: {:,}'.format(len(corpus)))
    print('  | # of sentences : {:,}'.format(len(list(itertools.chain(*[p.text.split('  ') for p in corpus])))))