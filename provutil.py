#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import datetime
import numpy as np
import pickle as pk
import pandas as pd
from copy import deepcopy


class ProvPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_model = os.path.sep.join((root, 'model'))
    fdir_result = os.path.sep.join((root, 'result'))


class ProvFunc(ProvPath):
    def normalize(self, text, do_lower=True, do_marking=True):
        if do_lower:
            text = deepcopy(text.lower())
        else:
            pass

        if do_marking:
            sent = text.split()
            for i, w in enumerate(sent):
                if re.match('www.', str(w)):
                    sent[i] = 'URL'
                elif re.search('\d+\d\.\d+', str(w)):
                    sent[i] = 'REF'
                elif re.match('\d', str(w)):
                    sent[i] = 'NUM'
                else:
                    continue

            text = deepcopy(' '.join(sent))
            del sent
        else:
            pass

        text = text.replace('?', '')
        text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)

        text = text.replace(' / ', '/')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')
        text = text.replace('\n', '  ')
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', ' ', text).strip()

        return text

    def save_corpus(self, corpus, fname_corpus):
        print('============================================================')
        print('Save corpus')

        fpath_corpus = os.path.join(self.fdir_corpus, fname_corpus)
        with open(fpath_corpus, 'wb') as f:
            pk.dump(corpus, f)

        print('  | fdir : {}'.format(self.fdir_corpus))
        print('  | fname: {}'.format(fname_corpus))

    def read_corpus(self, fname_corpus):
        fpath_corpus = os.path.join(self.fdir_corpus, fname_corpus)
        with open(fpath_corpus, 'rb') as f:
            corpus = pk.load(f)
        return corpus

    def save_result(self, result, fname_result):
        print('============================================================')
        print('Save result')

        fpath_result = os.path.join(self.fdir_result, fname_result)
        writer = pd.ExcelWriter(fpath_result)
        pd.DataFrame(result).to_excel(writer, 'Sheet1')
        writer.save()

        print('  | fdir : {}'.format(self.fdir_result))
        print('  | fname: {}'.format(fname_result))

    def encode_labels_binary(self, labels, target_label):
        labels_encoded = []
        for label_list in labels:
            if target_label in label_list:
                labels_encoded.append(1)
            else:
                labels_encoded.append(0)

        return labels_encoded


class ProvEval:
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)/len(labels_flat)