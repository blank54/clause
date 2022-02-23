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

import pandas as pd


def init_corpus(data_df):
    corpus, data_for_labeling = [], []
    for _, row in data_df.iterrows():
        for idx, sent in enumerate(row['text'].split('  ')):
            tag = '{}_{:02d}'.format(row['tag'], idx+1)
            clause = Doc(tag=tag, text=sent)
            corpus.append(clause)
            data_for_labeling.append('  TAGSPLIT  '.join((clause.tag, clause.text)))

    return corpus, data_for_labeling


if __name__ == '__main__':
    print('============================================================')
    print('Prepare data for labeling')

    ## Filenames
    fname_data = 'clause.xlsx'
    fname_data_for_labeling = 'sent_for_labeling.txt'
    fname_corpus = 'corpus.pk'

    ## Data to corpus
    print('--------------------------------------------------')
    print('Init corpus')

    fpath_data = os.path.sep.join((clausepath.fdir_data, fname_data))
    data_df = pd.read_excel(fpath_data)
    corpus, data_for_labeling = init_corpus(data_df)

    print('  | # of docs: {:,}'.format(len(corpus)))

    ## Save data for labeling
    print('--------------------------------------------------')
    print('Save data for labeling')

    fpath_data_for_labeling = os.path.sep.join((clausepath.fdir_data, fname_data_for_labeling))
    with open(fpath_data_for_labeling, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_for_labeling))

    print('  | fdir : {}'.format(clausepath.fdir_data))
    print('  | fname: {}'.format(fname_data_for_labeling))

    ## Save corpus
    print('--------------------------------------------------')
    print('Save corpus')

    clauseio.save_corpus(corpus=corpus, fname_corpus=fname_corpus)