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

from clauseutil import ClauseIO, ClausePath, ClauseModel
clauseio = ClauseIO()
clausepath = ClausePath()
clausemodel = ClauseModel()

import pickle as pk
import pandas as pd

import tensorflow as tf
from official.nlp import optimization

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def train(LABEL_NAME, parameters, fpath_model_checkpoint):
    print('============================================================')
    print('Main')

    print('--------------------------------------------------')
    print('  | Load dataset')

    train_ds, valid_ds, test_ds, class_names = clauseio.load_bert_dataset(LABEL_NAME, parameters)

    print('--------------------------------------------------')
    print('  | Model training')

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * parameters.get('EPOCHS')
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = parameters.get('LEARNING_RATE')
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    bert_preprocess, bert_encoder = clausemodel.bert_model_identification(bert_model_name=parameters.get('BERT_MODEL_NAME'))
    classifier_model = clausemodel.build_bert_model(ACTIVATION=parameters.get('ACTIVATION'),
                                                    tfhub_handle_preprocess=bert_preprocess,
                                                    tfhub_handle_encoder=bert_encoder,
                                                    additional=False)
    classifier_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                             metrics=['accuracy'])

    ## Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=fpath_model_checkpoint,
                                                          monitor='val_loss',
                                                          mode='min',
                                                          save_best_only=True,
                                                          save_weights_only=False)

    with tf.device('/device:GPU:2'):
        history = classifier_model.fit(x=train_ds,
                                       validation_data=valid_ds,
                                       epochs=parameters.get('EPOCHS'),
                                       callbacks=[early_stopping, model_checkpoint])

    return classifier_model, history

def save_model(model, history, fpath_bert_weights, fpath_bert_history):
    print('============================================================')
    print('Save model')

    classifier_model.save(fpath_bert_weights, include_optimizer=False)

    history_df = pd.DataFrame(history.history)
    with open(fpath_bert_history, 'w') as f:
        history_df.to_json(f)

    print(f'  | Save model weights: {fpath_bert_weights}')
    print(f'  | Save model history: {fpath_bert_history}')


if __name__ == '__main__':
    for model_version in ['5.4.0b', '5.4.1b']:
        parameters = clausemodel.set_parameters(model_version=model_version)

        label_list = clauseio.read_label_list(version='v6')
        # for LABEL_NAME in label_list:
        for LABEL_NAME in ['PROCEDURE', 'RnR', 'DEFINITION']:
            ## Initialize
            model_info, fpath_bert_weights, fpath_model_checkpoint, fpath_bert_history = clauseio.set_bert_filenames(LABEL_NAME, parameters)

            print('============================================================')
            print('Running information')
            print(f'  | Label name: {model_version}')
            print(f'  | Label name: {LABEL_NAME}')

            ## Train
            classifier_model, history = train(LABEL_NAME, parameters, fpath_model_checkpoint)
            save_model(classifier_model, history, fpath_bert_weights, fpath_bert_history)