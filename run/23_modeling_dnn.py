#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
https://www.tensorflow.org/tutorials/keras/text_classification?hl=ko
'''

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClausePath, ClauseModel, ClauseEval, ClauseFunc
clauseio = ClauseIO()
clausepath = ClausePath()
clausemodel = ClauseModel()
clauseeval = ClauseEval()
clausefunc = ClauseFunc()

import json
import itertools
import numpy as np
import pickle as pk
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def encode_labels(labels, LABEL_NAME):
    if LABEL_NAME in labels:
        return 1
    else:
        return 0

def get_word_index(corpus):
    word_index = {w:(v+3) for v, w in enumerate(list(set(itertools.chain(*[doc.normalized_text for doc in corpus]))))}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    return word_index

def data_preparation(corpus_dict, LABEL_NAME):
    global RANDOM_STATE, TRAIN_TEST_RATIO

    corpus = corpus_dict[LABEL_NAME]
    word_index = get_word_index(corpus)
    vocab_size = len(word_index)

    corpus_train, corpus_test = train_test_split(corpus, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)

    train_data = [[word_index[w] for w in doc.normalized_text] for doc in corpus_train]
    train_labels = [encode_labels(doc.labels, LABEL_NAME) for doc in corpus_train]

    test_data = [[word_index[w] for w in doc.normalized_text] for doc in corpus_test]
    y_test = [encode_labels(doc.labels, LABEL_NAME) for doc in corpus_test]

    train_ds = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

    x_test = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

    x_train, x_valid = train_test_split(train_ds, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)
    y_train, y_valid = train_test_split(train_labels, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)

    return vocab_size, x_train.tolist(), x_valid.tolist(), y_train, y_valid, x_test, y_test

def identify_model(vocab_size):
    classifier_model = keras.Sequential()
    classifier_model.add(keras.layers.Embedding(vocab_size, 100, input_shape=(None,)))
    classifier_model.add(keras.layers.GlobalAveragePooling1D())
    classifier_model.add(keras.layers.Dense(16, activation='sigmoid'))
    classifier_model.add(keras.layers.Dropout(0.1))
    classifier_model.add(keras.layers.Dense(16, activation='sigmoid'))
    classifier_model.add(keras.layers.Dropout(0.1))
    classifier_model.add(keras.layers.Dense(1, activation='sigmoid'))

    return classifier_model

def train(vocab_size, x_train, y_train, x_valid, y_valid):
    classifier_model = identify_model(vocab_size)
    classifier_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = classifier_model.fit(x_train,
                                   y_train,
                                   epochs=100,
                                   batch_size=8,
                                   validation_data=(x_valid, y_valid),
                                   verbose=1)

    return classifier_model, history

def save_model(model, history, fpath_dnn_weights, fpath_dnn_history):
    print('============================================================')
    print('Save model')

    classifier_model.save(fpath_dnn_weights, include_optimizer=False)

    history_df = pd.DataFrame(history.history)
    with open(fpath_dnn_history, 'w') as f:
        history_df.to_json(f)

    print(f'  | Save model weights: {fpath_dnn_weights}')
    print(f'  | Save model history: {fpath_dnn_history}')

def load_model(vocab_size, fpath_dnn_weights):
    reloaded_model = identify_model(vocab_size)
    reloaded_model.load_weights(fpath_dnn_weights)

    return reloaded_model

def loss_curve(history, fname):
    loss = history['loss'].values()
    val_loss = history['val_loss'].values()
    epochs = range(1, len(loss) + 1)
    
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss curve')
    plt.ylabel('Loss')
    plt.legend()
    
    fpath = os.path.sep.join((clausepath.fdir_result, fname))
    plt.savefig(fpath, dpi=600)

def acc_curve(history, fname):
    acc = history['accuracy'].values()
    val_acc = history['val_accuracy'].values()
    epochs = range(1, len(acc) + 1)
    
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy curve')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    fpath = os.path.sep.join((clausepath.fdir_result, fname))
    plt.savefig(fpath, dpi=600)

def evaluate(model, x_test, y_test):
    y_pred = clauseeval.get_preds(model=model, dataset=x_test)
    y_labels = y_test

    model_evaluation_result = {
        'confusion': confusion_matrix(y_labels, y_pred),
        'precision': precision_score(y_labels, y_pred),
        'recall': recall_score(y_labels, y_pred),
        'f1': f1_score(y_labels, y_pred),
    }

    return model_evaluation_result

def show_results(results):
    for LABEL_NAME, result in results.items():
        print('--------------------------------------------------')
        print(f'Label: {LABEL_NAME}')
            
        try:
            print('  | Precision : {:.03f}'.format(result['precision']))
            print('  | Recall    : {:.03f}'.format(result['recall']))
            print('  | F1 score  : {:.03f}'.format(result['f1']))
            print('  | Confusion : {}'.format(result['confusion']))
        except:
            print('  | No results ...')


if __name__ == '__main__':
    ## Filenames
    fname_corpus = 'corpus_sampled.pk'

    ## Parameters
    do_train = True

    RANDOM_STATE = 42
    TRAIN_TEST_RATIO = 0.8

    ## Corpus
    corpus_dict = clauseio.read_corpus(fname_corpus=fname_corpus)
    label_list = clauseio.read_label_list(version='v7')
    results = defaultdict(dict)

    ## Train
    for LABEL_NAME in label_list:
        print('============================================================')
        print('Running information')
        print(f'  | Label name: {LABEL_NAME}')

        ## Filenames
        fpath_dnn_weights = f'dnn_weights_{LABEL_NAME}.h5'
        fpath_dnn_history = f'dnn_history_{LABEL_NAME}.json'

        ## Data preparation
        vocab_size, x_train, x_valid, y_train, y_valid, x_test, y_test = data_preparation(corpus_dict, LABEL_NAME)

        ## Train
        if do_train:
            classifier_model, history = train(vocab_size, x_train, y_train, x_valid, y_valid)
            save_model(classifier_model, history, fpath_dnn_weights, fpath_dnn_history)
        else:
            pass

        ## Evaluate
        try:
            reloaded_model = load_model(vocab_size, fpath_dnn_weights)
            results[LABEL_NAME] = evaluate(model=reloaded_model, x_test=x_test, y_test=y_test)

            with open(fpath_dnn_history, 'r') as f:
                history = json.load(f)

            fname_loss = f'loss_dnn_{LABEL_NAME}.png'
            fname_acc = f'acc_dnn_{LABEL_NAME}.png'
            loss_curve(history=history, fname=fname_loss)
            acc_curve(history=history, fname=fname_acc)
        except OSError:
            results[LABEL_NAME] = {}

        print('--------------------------------------------------')
        print('Show results')

        show_results(results)

    print('--------------------------------------------------')
    print('Save results')

    fname_result = f'result_dnn.xlsx'
    clauseio.save_result(results, fname_result)