#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clutil import ClPath
clpath = ClPath()

import pandas as pd
import pickle as pk
from collections import defaultdict


def build_corpus(fname_data):
    fpath_data = os.path.join(clpath.fdir_data, fname_data)

    corpus = []
    df = pd.read_excel(fpath_data)
    for idx, row in df.iterrows():
        clause = Doc(tag=row['tag'], text=row['text'])
        corpus.append(clause)
    return corpus

def save_corpus(corpus, fname_corpus):
    fpath_corpus = os.path.join(clpath.fdir_corpus, fname_corpus)
    with open(fpath_corpus, 'wb') as f:
        pk.dump(corpus, f)

    print('Build corpus')
    print('  | fdir : {}'.format(clpath.fdir_corpus))
    print('  | fname: {}'.format(fname_corpus))


if __name__ == '__main__':
    fname_data = 'clause.xlsx'
    fname_corpus = 'clause.pk'

    corpus = build_corpus(fname_data=fname_data)
    save_corpus(corpus=corpus, fname_corpus=fname_corpus)