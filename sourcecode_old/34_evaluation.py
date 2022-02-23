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
    reloaded_model = clausemodel.build_bert_model(ACTIVATION_hidden=parameters.get('ACTIVATION_hidden'),
                                                  ACTIVATION_output=parameters.get('ACTIVATION_output'),
                                                  tfhub_handle_preprocess=bert_preprocess,
                                                  tfhub_handle_encoder=bert_encoder)
    reloaded_model.load_weights(fpath_bert_weights)

    return reloaded_model

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

    model_version = '5.3.0a'

    parameters = clausemodel.set_parameters(model_version=model_version)
    results = defaultdict(dict)

    label_list = clauseio.read_label_list(version='v6')
    for LABEL_NAME in label_list:
        print(f'  | Label name: {LABEL_NAME}')

        model_info, fpath_bert_weights, fpath_bert_history = clauseio.set_bert_filenames(LABEL_NAME, parameters)
        # with open(fpath_bert_history, 'rb') as f:
        #     history = pk.load(f)

        # history_dict = history.history
        # acc = history_dict['binary_accuracy']
        # val_acc = history_dict['val_binary_accuracy']
        # loss = history_dict['loss']
        # val_loss = history_dict['val_loss']

        # epochs = range(1, len(acc) + 1)
        # fig = plt.figure(figsize=(10, 6))
        # fig.tight_layout()

        # plt.subplot(2, 1, 1)
        # # r is for "solid red line"
        # plt.plot(epochs, loss, 'r', label='Training loss')
        # # b is for "solid blue line"
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.title('Training and validation loss')
        # # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()

        # plt.subplot(2, 1, 2)
        # plt.plot(epochs, acc, 'r', label='Training acc')
        # plt.plot(epochs, val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='lower right')


        try:
            ## Initialize
            model_info, fpath_bert_weights, fpath_bert_history = clauseio.set_bert_filenames(LABEL_NAME, parameters)

            ## Load model
            reloaded_model = load(parameters=parameters, fpath_bert_weights=fpath_bert_weights)

            ## Load dataset
            _, _, test_ds, _ = clauseio.load_bert_dataset(LABEL_NAME, parameters)

            ## Evaluate
            results[LABEL_NAME] = evaluate(model=reloaded_model, test_ds=test_ds)
        except OSError:
            results[LABEL_NAME] = {}

    print('--------------------------------------------------')
    print('Show results')

    show_results(results)

    print('--------------------------------------------------')
    print('Show results')

    fname_result = f'result_{model_version}.xlsx'
    clauseio.save_result(results, fname_result)