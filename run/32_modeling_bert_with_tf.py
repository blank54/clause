#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
https://www.tensorflow.org/text/tutorials/classify_text_with_bert
'''


# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClausePath, ClauseFunc
clauseio = ClauseIO()
clausepath = ClausePath()
clausefunc = ClauseFunc()

import shutil
import numpy as np
from collections import defaultdict

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def fdir(label_name, option, **kwargs):
    sampling = kwargs.get('sampling', '')

    if all([(option == 'train'), sampling]):
        return os.path.sep.join((clausepath.fdir_data, sampling, label_name, option))
    else:
        return os.path.sep.join((clausepath.fdir_data, label_name, option))


def load_dataset(label_name):
    global AUTOTUNE, BATCH_SIZE, TRAIN_VALID_RATIO, RANDOM_STATE

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        fdir(label_name=label_name, option='train', sampling='downsampled'),
        batch_size=BATCH_SIZE,
        validation_split=(1-TRAIN_VALID_RATIO),
        subset='training',
        seed=RANDOM_STATE)
    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    valid_ds = tf.keras.utils.text_dataset_from_directory(
        fdir(label_name=label_name, option='train', sampling='downsampled'),
        batch_size=BATCH_SIZE,
        validation_split=(1-TRAIN_VALID_RATIO),
        subset='validation',
        seed=RANDOM_STATE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        fdir(label_name=label_name, option='test', sampling='downsampled'),
        batch_size=BATCH_SIZE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names

def build_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    global ACTIVATION_hidden, ACTIVATION_output

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)

    ## Additional layers
    net = tf.keras.layers.Dense(64, activation=ACTIVATION_hidden)(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(32, activation=ACTIVATION_hidden)(net)
    net = tf.keras.layers.Dropout(0.1)(net)

    net = tf.keras.layers.Dense(1, activation=ACTIVATION_output, name='classifier')(net)

    return tf.keras.Model(text_input, net)

def model_identification(bert_model_name):
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
    model = build_model(tfhub_handle_preprocess, tfhub_handle_encoder)

    return model

def train(train_ds, valid_ds):
    global BERT_MODEL_NAME, EPOCHS, LEARNING_RATE

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = LEARNING_RATE
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    classifier_model = model_identification(bert_model_name=BERT_MODEL_NAME)
    classifier_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                             metrics=['accuracy'])

    with tf.device('/device:GPU:2'):
        history = classifier_model.fit(x=train_ds,
                                       validation_data=valid_ds,
                                       epochs=EPOCHS)

    return classifier_model, history

def get_labels(ds):
    y_labels = []
    for batch, labels in ds:
        y_labels.extend([l.numpy() for l in labels])
    return y_labels

def get_preds(model, ds):
    y_preds = []
    for pred in model.predict(ds):
        if pred >= 0.5:
            y_preds.append(1)
        else:
            y_preds.append(0)
    return y_preds


if __name__ == '__main__':
    for BATCH_SIZE in [8, 16]:
        for LEARNING_RATE in [2e-5, 3e-5, 2e-4]:
            results = defaultdict(list)
            label_list = clauseio.read_label_list(version='v2')
            for LABEL_NAME in label_list:
                ## Parameters
                print('============================================================')
                print('Running information')

                AUTOTUNE = tf.data.AUTOTUNE
                # BERT_MODEL_NAME = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
                BERT_MODEL_NAME = 'bert_en_cased_L-12_H-768_A-12'

                RANDOM_STATE = 42
                TRAIN_VALID_RATIO = 0.75
                EPOCHS = 10

                BATCH_SIZE = 16
                ACTIVATION_hidden = 'relu'
                ACTIVATION_output = 'sigmoid'
                # LEARNING_RATE = 2e-5

                print(f'  | Label name: {LABEL_NAME}')
                print(f'  | BERT model: {BERT_MODEL_NAME}')

                ## Filenames
                base = '1,976'
                model_version = '5.2.0'
                model_info = f'V-{model_version}_D-{base}_BS-{BATCH_SIZE}_EP-{EPOCHS}_LR-{LEARNING_RATE}'

                ## Device setting
                print('============================================================')
                print('GPU allocation')

                DEVICE = clausefunc.gpu_allocation()

                ## Model train
                print('============================================================')
                print('Main')

                print('--------------------------------------------------')
                print('  | Load dataset')

                train_ds, valid_ds, test_ds, class_names = load_dataset(label_name=LABEL_NAME)

                print('--------------------------------------------------')
                print('  | Model training')

                classifier_model, history = train(train_ds=train_ds, valid_ds=valid_ds)
                loss, accuracy = classifier_model.evaluate(test_ds)

                ## Save & load
                print('============================================================')
                print('Save and load model')

                print('--------------------------------------------------')

                fname_bert_weights = f'bert_weights-{model_info}_LB-{LABEL_NAME}.h5'
                fpath_bert_weights = os.path.sep.join((clausepath.fdir_model, fname_bert_weights))
                classifier_model.save(fpath_bert_weights, include_optimizer=False)

                reloaded_model = model_identification(bert_model_name=BERT_MODEL_NAME)
                reloaded_model.load_weights(fpath_bert_weights)

                print(f'  | Save model weights: {fname_bert_weights}')

                ## Evaluate
                print('============================================================')
                print('Evaluation')
                
                y_pred = get_preds(model=reloaded_model, ds=test_ds)
                y_labels = get_labels(ds=test_ds)

                confusion = confusion_matrix(y_labels, y_pred)
                precision = precision_score(y_labels, y_pred)
                recall = recall_score(y_labels, y_pred)
                f1 = f1_score(y_labels, y_pred)

                print(f'  | Label name: {LABEL_NAME}')
                print(f'  | Loss      : {loss:.03f}')
                print(f'  | Accuracy  : {accuracy:.03f}')
                print(f'  | Precision : {precision:.03f}')
                print(f'  | Recall    : {recall:.03f}')
                print(f'  | F1 score  : {f1:.03f}')
                print(f'  | Confusion : {confusion}')

                ## Results
                results['label'].append(LABEL_NAME)
                results['loss'].append(loss)
                results['accuracy'].append(accuracy)
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['f1'].append(f1)

            print('============================================================')
            print('Save results')

            fname_result = f'result_{model_version}.xlsx'
            clauseio.save_result(results, fname_result)