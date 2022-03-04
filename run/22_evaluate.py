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

from clauseutil import ClauseIO, ClausePath, ClauseModel, ClauseEval
clauseio = ClauseIO()
clausepath = ClausePath()
clausemodel = ClauseModel()
clauseeval = ClauseEval()

import json
import pickle as pk
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def load(parameters, fpath_bert_weights):
    bert_preprocess, bert_encoder = clausemodel.bert_model_identification(bert_model_name=parameters.get('BERT_MODEL_NAME'))
    reloaded_model = clausemodel.build_bert_model(ACTIVATION=parameters.get('ACTIVATION'),
                                                  tfhub_handle_preprocess=bert_preprocess,
                                                  tfhub_handle_encoder=bert_encoder,
                                                  additional=False)
    reloaded_model.load_weights(fpath_bert_weights)

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

def evaluate(model, test_ds):
    y_pred = clauseeval.get_preds(model=model, dataset=test_ds)
    y_labels = clauseeval.get_labels(dataset=test_ds)

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
    print('============================================================')
    print('BERT model evaluation')

    print('--------------------------------------------------')
    print('Load model')

    for model_version in ['6.0.0', '6.0.1']:
        parameters = clausemodel.set_parameters(model_version=model_version)
        results = defaultdict(dict)

        label_list = clauseio.read_label_list(version='v7')
        for LABEL_NAME in label_list:
        # for LABEL_NAME in ['DEFINITION']:
            print(f'  | Label name: {LABEL_NAME}')

            try:
                model_info, fpath_bert_weights, fpath_model_checkpoint, fpath_bert_history = clauseio.set_bert_filenames(LABEL_NAME, parameters)
                _, _, test_ds, _ = clauseio.load_bert_dataset(LABEL_NAME, parameters)

                ## Load model
                # reloaded_model = load(parameters=parameters, fpath_bert_weights=fpath_bert_weights)
                reloaded_model = load(parameters=parameters, fpath_bert_weights=fpath_model_checkpoint)
                results[LABEL_NAME] = evaluate(model=reloaded_model, test_ds=test_ds)

                ## Learning curve
                with open(fpath_bert_history, 'r') as f:
                    history = json.load(f)

                fname_loss = f'loss_{model_version}_{LABEL_NAME}.png'
                fname_acc = f'acc_{model_version}_{LABEL_NAME}.png'
                loss_curve(history=history, fname=fname_loss)
                acc_curve(history=history, fname=fname_acc)

            except OSError:
                results[LABEL_NAME] = {}

        print('--------------------------------------------------')
        print('Show results')

        show_results(results)

        print('--------------------------------------------------')
        print('Show results')

        fname_result = f'result_{model_version}.xlsx'
        clauseio.save_result(results, fname_result)