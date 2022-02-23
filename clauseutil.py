#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import numpy as np
import pickle as pk
import pandas as pd
from collections import defaultdict

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class ClausePath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_thesaurus = os.path.sep.join((root, 'thesaurus'))
    fdir_model = os.path.sep.join((root, 'model'))
    fdir_result = os.path.sep.join((root, 'result'))
    fdir_checkpoint = os.path.sep.join((fdir_model, 'checkpoint'))


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

    def save_corpus_sheet(self, corpus_sheet, fname):
        fpath_result = os.path.sep.join((self.fdir_corpus, fname))
        writer = pd.ExcelWriter(fpath_result)
        pd.DataFrame(corpus_sheet).to_excel(writer, 'Sheet1')
        writer.save()

        print('  | fdir : {}'.format(self.fdir_corpus))
        print('  | fname: {}'.format(fname))

    def read_corpus_sheet(self, fname):
        fpath_result = os.path.sep.join((self.fdir_corpus, fname))
        result = pd.read_excel(fpath_result)
        return result

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

    def read_label_list(self, version):
        fname_label_list = 'label_list_{}.txt'.format(version)
        fpath_label_list = os.path.sep.join((self.fdir_corpus, fname_label_list))
        with open(fpath_label_list, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.read().strip().split('\n')]

    def set_bert_filenames(self, LABEL_NAME, parameters):
        model_version = parameters.get('model_version')
        base = parameters.get('base')
        BATCH_SIZE = parameters.get('BATCH_SIZE')
        EPOCHS = parameters.get('EPOCHS')
        LEARNING_RATE = parameters.get('LEARNING_RATE')

        model_info = f'V-{model_version}_D-{base}_BS-{BATCH_SIZE}_EP-{EPOCHS}_LR-{LEARNING_RATE}'

        fname_bert_weights = f'bert_weights-{model_info}_LB-{LABEL_NAME}.h5'
        fpath_bert_weights = os.path.sep.join((self.fdir_model, fname_bert_weights))

        fname_model_checkpoint = f'bert_cp_{model_version}_{LABEL_NAME}.h5'
        fpath_model_checkpoint = os.path.sep.join((self.fdir_checkpoint, fname_model_checkpoint))

        fname_bert_history = f'bert_history-{model_info}_LB-{LABEL_NAME}.json'
        fpath_bert_history = os.path.sep.join((self.fdir_model, fname_bert_history))

        return model_info, fpath_bert_weights, fpath_model_checkpoint, fpath_bert_history

    def load_bert_dataset(self, LABEL_NAME, parameters):
        fdir_train_ds = os.path.sep.join((self.fdir_data, LABEL_NAME, 'train'))
        fdir_test_ds = os.path.sep.join((self.fdir_data, LABEL_NAME, 'test'))

        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            fdir_train_ds,
            batch_size=parameters.get('BATCH_SIZE'),
            validation_split=(1-parameters.get('TRAIN_VALID_RATIO')),
            subset='training',
            seed=parameters.get('RANDOM_STATE'))
        class_names = raw_train_ds.class_names
        train_ds = raw_train_ds.cache().prefetch(buffer_size=parameters.get('AUTOTUNE'))

        valid_ds = tf.keras.utils.text_dataset_from_directory(
            fdir_train_ds,
            batch_size=parameters.get('BATCH_SIZE'),
            validation_split=(1-parameters.get('TRAIN_VALID_RATIO')),
            subset='validation',
            seed=parameters.get('RANDOM_STATE'))
        valid_ds = valid_ds.cache().prefetch(buffer_size=parameters.get('AUTOTUNE'))

        test_ds = tf.keras.utils.text_dataset_from_directory(
            fdir_test_ds,
            batch_size=parameters.get('BATCH_SIZE'))
        test_ds = test_ds.cache().prefetch(buffer_size=parameters.get('AUTOTUNE'))

        return train_ds, valid_ds, test_ds, class_names


class ClauseFunc:
    def encode_labels_binary(self, labels, target_label):
        labels_encoded = []
        for label_list in labels:
            if target_label in label_list:
                labels_encoded.append(1)
            else:
                labels_encoded.append(0)

        return labels_encoded

    def encode_labels_multi(self, labels, label_list):
        labels_encoded = []
        for label in labels:
            label_code = '0000000000'
            label_code = list(label_code)
            for l in label:
                label_code[label_list.index(l)] = '1'
                
            label_code = ''.join(label_code)
            label_value = sum(int(e)*(2**i) for i, e in enumerate(list(label_code)))
            labels_encoded.append(label_value)

        return labels_encoded

    def build_dataloader(self, inputs, labels, masks, batch_size, encode, **kwargs):
        target_label = kwargs.get('target_label')

        inputs = torch.tensor(inputs)    
        masks = torch.tensor(masks)
        
        if encode == 'binary':
            labels = torch.tensor(self.encode_labels_binary(labels=labels, target_label=target_label))
        elif encode == 'multi':
            label_list = ClauseIO().read_label_list(version='v2')
            labels = torch.tensor(self.encode_labels_multi(labels=labels, label_list=label_list))
        else:
            labels = torch.tensor(labels)

        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)

        return DataLoader(data, sampler=sampler, batch_size=batch_size)

    def gpu_allocation(self):
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


class ClauseModel(ClausePath):
    def version_map(self):
        version_dict = defaultdict(dict)
        fname_bert_model_version_map = 'bert_model_map.txt'
        fpath_bert_model_version_map = os.path.sep.join((self.fdir_corpus, fname_bert_model_version_map))
        with open(fpath_bert_model_version_map, 'r', encoding='utf-8') as f:
            for row in f.read().strip().split('\n'):
                version, BATCH_SIZE, EPOCHS, LEARNING_RATE, ACTIVATION = row.strip().split('  ')
                version_dict[version]['BATCH_SIZE'] = int(BATCH_SIZE)
                version_dict[version]['EPOCHS'] = int(EPOCHS)
                version_dict[version]['LEARNING_RATE'] = float(LEARNING_RATE)
                version_dict[version]['ACTIVATION'] = ACTIVATION

        return version_dict

    def set_parameters(self, model_version):
        version_dict = self.version_map()
        parameters = {
            'model_version': model_version,
            'BERT_MODEL_NAME': 'bert_en_cased_L-12_H-768_A-12',
            # 'BERT_MODEL_NAME': 'small_bert/bert_en_uncased_L-4_H-512_A-8',

            'AUTOTUNE': tf.data.AUTOTUNE,
            'RANDOM_STATE': 42,
            'TRAIN_VALID_RATIO': 0.75,

            'BATCH_SIZE': version_dict[model_version]['BATCH_SIZE'],
            'LEARNING_RATE': version_dict[model_version]['LEARNING_RATE'],
            'EPOCHS': version_dict[model_version]['EPOCHS'],
            'ACTIVATION': version_dict[model_version]['ACTIVATION'],
            }

        return parameters

    def bert_model_identification(self, bert_model_name):
        map_name_to_handle = {
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        }

        map_model_to_preprocess = {
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }

        tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
        tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        return tfhub_handle_preprocess, tfhub_handle_encoder

    def build_bert_model(self, ACTIVATION, tfhub_handle_preprocess, tfhub_handle_encoder, additional):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)

        ## Additional layers
        if additional:
            net = tf.keras.layers.Dense(64, activation=ACTIVATION)(net)
            net = tf.keras.layers.Dropout(0.1)(net)
            net = tf.keras.layers.Dense(32, activation=ACTIVATION)(net)
            net = tf.keras.layers.Dropout(0.1)(net)
        else:
            pass

        net = tf.keras.layers.Dense(1, activation=ACTIVATION, name='classifier')(net)

        return tf.keras.Model(text_input, net)


class ClauseEval:
    def label_distribution(self, corpus, **kwargs):
        show_only = kwargs.get('show_only', '')

        total = len(corpus)
        label_cnt = defaultdict(int)
        for doc in corpus:
            for LABEL_NAME in doc.labels:
                label_cnt[LABEL_NAME] += 1

        if show_only:
            cnt = label_cnt[show_only]
            print(f'  | {show_only:10}: POS-{cnt:<6,} NEG-{total-cnt:6,}')
        else:
            for LABEL_NAME, cnt in label_cnt.items():
                print(f'  | {LABEL_NAME:10}: POS-{cnt:<6,} NEG-{total-cnt:6,}')

        return label_cnt

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat)/len(labels_flat)

    def get_labels(self, dataset):
        y_labels = []
        for batch, labels in dataset:
            y_labels.extend([l.numpy() for l in labels])
        return y_labels

    def get_preds(self, model, dataset):
        y_preds = []
        for pred in model.predict(dataset):
            if pred >= 0.5:
                y_preds.append(1)
            else:
                y_preds.append(0)
        return y_preds