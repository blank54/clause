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

import pickle as pk


def read_corpus(fname_corpus):
    fpath_corpus = os.path.join(provpath.fdir_corpus, fname_corpus)
    with open(fpath_corpus, 'rb') as f:
        corpus = pk.load(f)
    return corpus

def convert_corpus(corpus):
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


if __name__ == '__main__':
    fname_corpus = 'provision.pk'
    corpus = read_corpus(fname_corpus=fname_corpus)
    convert_corpus(corpus=corpus)