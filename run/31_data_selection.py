#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
https://www.tensorflow.org/text/tutorials/classify_text_with_bert
https://precommer.tistory.com/m/47
'''

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClauseModel, ClauseEval
clauseio = ClauseIO()
clausemodel = ClauseModel()
clauseeval = ClauseEval()

import pickle as pk
import itertools
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def load(parameters, fpath_bert_weights, LABEL_NAME):
    bert_preprocess, bert_encoder = clausemodel.bert_model_identification(bert_model_name=parameters.get('BERT_MODEL_NAME'))
    reloaded_model = clausemodel.build_bert_model(ACTIVATION=parameters.get('ACTIVATION'),
                                                  tfhub_handle_preprocess=bert_preprocess,
                                                  tfhub_handle_encoder=bert_encoder,
                                                  additional=False)
    reloaded_model.load_weights(fpath_bert_weights)

    _, _, test_ds, _ = clauseio.load_bert_dataset(LABEL_NAME, parameters)

    return reloaded_model, test_ds

def predict(model, dataset):
    y_pred = clauseeval.get_preds(model=model, dataset=dataset)

    y_texts = []
    y_labels = []
    for batch, labels in dataset:
        y_texts.extend([b.numpy().decode('utf-8') for b in batch])
        y_labels.extend([l.numpy() for l in labels])

    return y_pred, y_labels, y_texts

def select(y_pred, y_labels, y_texts):
    target_texts = []
    for label, pred, text in zip(y_labels, y_pred, y_texts):
        if label == 0 and pred == 1:
            target_texts.append(text)

    target_tags = []
    for text in target_texts:
        for doc in corpus:
            if text == ' '.join(doc.normalized_text):
                target_tags.append(doc.tag)

    return target_tags


if __name__ == '__main__':
    print('============================================================')
    print('Select data')

    ## Corpus
    print('--------------------------------------------------')
    print('Load corpus')

    fname_corpus = 'corpus_sampled.pk'
    fname_corpus_selected = 'corpus_selected.pk'

    ## Model
    print('--------------------------------------------------')
    print('Load model')

    model_version = '5.4.1s'
    parameters = clausemodel.set_parameters(model_version=model_version)

    target_tags = {}
    # label_list = clauseio.read_label_list(version='v6')
    label_list = ['CONDITION']
    for LABEL_NAME in label_list:
        corpus = clauseio.read_corpus(fname_corpus)[LABEL_NAME]
        print('  | # of docs: {:,}'.format(len(corpus)))

        model_info, fpath_bert_weights, fpath_model_checkpoint, fpath_bert_history = clauseio.set_bert_filenames(LABEL_NAME, parameters)
        reloaded_model, test_ds = load(parameters, fpath_bert_weights, LABEL_NAME)
        y_pred, y_labels, y_texts = predict(reloaded_model, test_ds)

        print('  | Model version: {}'.format(model_version))
        print('  | Label name   : {}'.format(LABEL_NAME))
        print('  | y_pred       : {:,}'.format(len(y_pred)))
        print('  | y_labels     : {:,}'.format(len(y_labels)))
        print('  | y_texts      : {:,}'.format(len(y_texts)))

        ## Select targets
        target_tags[LABEL_NAME] = select(y_pred, y_labels, y_texts)

    print('--------------------------------------------------')
    print('Select targets')

    for LABEL_NAME in target_tags.keys():
        print('  | {}: {:,}'.format(LABEL_NAME, len(target_tags[LABEL_NAME])))

    ## Remove targets from corpus
    print('--------------------------------------------------')
    print('Remove targets from corpus')

    target_tags_integrated = list(itertools.chain(*[tags for LABEL_NAME, tags in target_tags.items()]))
    corpus_selected = [doc for doc in corpus if doc.tag not in target_tags_integrated]

    print('  | # of corpus  : {:,}'.format(len(corpus)))
    print('  | # of selected: {:,}'.format(len(corpus_selected)))

    ## Save selected corpus
    print('--------------------------------------------------')
    print('Save selected corpus')

    clauseio.save_corpus(corpus_selected, fname_corpus_selected)