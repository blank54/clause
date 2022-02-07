#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClausePath
clauseio = ClauseIO()
clausepath = ClausePath()

import random
import shutil
from tqdm import tqdm


def make_directories(label_name):
    os.makedirs(os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'train', 'pos')), exist_ok=True)
    os.makedirs(os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'train', 'neg')), exist_ok=True)
    os.makedirs(os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'test', 'pos')), exist_ok=True)
    os.makedirs(os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'test', 'neg')), exist_ok=True)

def down_sampling(label_name):
    fdir_tr_pos = os.path.sep.join((clausepath.fdir_data, label_name, 'train', 'pos'))
    fdir_tr_neg = os.path.sep.join((clausepath.fdir_data, label_name, 'train', 'neg'))

    flist_tr_pos = os.listdir(fdir_tr_pos)
    flist_tr_neg = os.listdir(fdir_tr_neg)

    cnt_pos = len(flist_tr_pos)
    cnt_neg = len(flist_tr_neg)
    cnt_pivot = int(min(cnt_pos, cnt_neg)*SAMPLING_RATIO)

    try:
        flist_pos = random.sample(flist_tr_pos, cnt_pivot)
    except ValueError:
        flist_pos = flist_tr_pos

    try:
        flist_neg = random.sample(flist_tr_neg, cnt_pivot)
    except ValueError:
        flist_neg = flist_tr_neg

    print(f'  | POS in train set: {cnt_pos:,d}')
    print(f'  | NEG in train set: {cnt_neg:,d}')
    print(f'  | Sample size     : {cnt_pivot:,d}')

    return flist_pos, flist_neg

def save_downsampled_data(flist, option):
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            fpath_origin = os.path.sep.join((clausepath.fdir_data, label_name, 'train', option, fname))
            fpath_destin = os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'train', option, fname))
        
            shutil.copy(fpath_origin, fpath_destin)
            pbar.update(1)

    # print(f'  | {option} in downsampled data: {len(os.listdir(os.path.sep.join((clausepath.fdir_data, 'downsampled', label_name, 'train', option))))}')

if __name__ == '__main__':
    ## Filenames
    base = '1,976'

    ## Parameters
    SAMPLING_RATIO = 1.5

    label_list = clauseio.read_label_list(version='v2')
    for label_name in label_list:
        print('============================================================')
        print(f'Downsampling on [{label_name}]')

        flist_pos, flist_neg = down_sampling(label_name)
        make_directories(label_name=label_name)
        save_downsampled_data(flist=flist_pos, option='pos')
        save_downsampled_data(flist=flist_neg, option='neg')