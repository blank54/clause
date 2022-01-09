#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import numpy as np
import pickle as pk
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class ClausePath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_thesaurus = os.path.sep.join((root, 'thesaurus'))
    fdir_model = os.path.sep.join((root, 'model'))
    fdir_result = os.path.sep.join((root, 'result'))


class ClauseIO(ClausePath):
    def save_corpus(self, corpus, fname_corpus):
        fpath_corpus = os.path.sep.join((self.fdir_corpus, fname_corpus))
        with open(fpath_corpus, 'wb') as f:
            pk.dump(corpus, f)

        print('  | fdir : {}'.format(self.fdir_corpus))
        print('  | fname: {}'.format(fname_corpus))

    def read_corpus(self, fname_corpus):
        fpath_corpus = os.path.sep.join((self.fdir_corpus, fname_corpus))
        with open(fpath_corpus, 'rb') as f:
            corpus = pk.load(f)

        return corpus

    def save_model(self, model, fname_model):
        fpath_model = os.path.sep.join((self.fdir_model, fname_model))
        with open(fpath_model, 'wb') as f:
            pk.dump(model, f)

        print('  | fdir : {}'.format(self.fdir_model))
        print('  | fname: {}'.format(fname_model))

    def read_model(self, fname_model):
        fpath_model = os.path.sep.join((self.fdir_model, fname_model))
        with open(fpath_model, 'rb') as f:
            model = pk.load(f)
        return model

    def save_result(self, result, fname_result):
        print('============================================================')
        print('Save result')

        fpath_result = os.path.sep.join((self.fdir_result, fname_result))
        writer = pd.ExcelWriter(fpath_result)
        pd.DataFrame(result).to_excel(writer, 'Sheet1')
        writer.save()

        print('  | fdir : {}'.format(self.fdir_result))
        print('  | fname: {}'.format(fname_result))

    def read_result(self, fname_result):
        fpath_result = os.path.sep.join((self.fdir_result, fname_result))
        result = pd.read_excel(fpath_result)
        return result

    def argv2bool(self, argv):
        if any(((argv=='t'), (argv=='true'), (argv=='True'))):
            return True
        elif any(((argv=='f'), (argv=='false'), (argv=='False'))):
            return False
        else:
            print('ArgvError: Wrong argv')
            sys.exit()


class ClauseFunc:
    def encode_labels_binary(self, labels, target_label):
        labels_encoded = []
        for label_list in labels:
            if target_label in label_list:
                labels_encoded.append(1)
            else:
                labels_encoded.append(0)

        return labels_encoded

    def build_dataloader(self, inputs, labels, masks, batch_size, target_label, encode):
        inputs = torch.tensor(inputs)    
        masks = torch.tensor(masks)
        
        if encode:
            labels = torch.tensor(self.encode_labels_binary(labels=labels, target_label=target_label))
        else:
            labels = torch.tensor(labels)

        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)

        return DataLoader(data, sampler=sampler, batch_size=batch_size)

    def gpu_allocation(self):
        print('============================================================')
        print('GPU allocation')

        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']= '0'

        if torch.cuda.is_available():    
            device = torch.device('cuda')
            print('  | There are {} GPU(s) available.'.format(torch.cuda.device_count()))
            print('  | Current CUDA device: {}'.format(torch.cuda.current_device()))
        else:
            device = torch.device('cpu')
            print('  | No GPU available, using the CPU instead.')

        return device



class ClauseEval:
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat)/len(labels_flat)