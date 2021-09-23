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

import pandas as pd
import pickle as pk
from collections import defaultdict


def build_corpus(fname_data):
    fpath_data = os.path.join(provpath.fdir_data, fname_data)

    corpus = []
    df = pd.read_excel(fpath_data)

    print('Build corpus')
    for idx, row in df.iterrows():
        provision = Doc(tag=row['tag'], text=row['text'], normalized_text=provfunc.normalize(text=row['text']))
        corpus.append(provision)
        print('\r  | Normalization: {:,d}'.format(idx+1), end='')
    print('\n  | Total {:,d} provisions'.format(len(corpus)))
    return corpus

def save_corpus(corpus, fname_corpus):
    fpath_corpus = os.path.join(provpath.fdir_corpus, fname_corpus)
    with open(fpath_corpus, 'wb') as f:
        pk.dump(corpus, f)

    print('Save corpus')
    print('  | fdir : {}'.format(provpath.fdir_corpus))
    print('  | fname: {}'.format(fname_corpus))


if __name__ == '__main__':
    fname_data = 'provision.xlsx'
    fname_corpus = 'provision.pk'

    corpus = build_corpus(fname_data=fname_data)
    save_corpus(corpus=corpus, fname_corpus=fname_corpus)