#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
http://yonghee.io/bert_binary_classification_naver/
https://nomalcy.tistory.com/211
'''

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClauseFunc, ClauseEval
clauseio = ClauseIO()
clausefunc = ClauseFunc()
clauseeval = ClauseEval()

import torch
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer

import random
import numpy as np
import pickle as pk
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix

def model_training(train_dataloader, valid_dataloader):
    global RANDOM_STATE, EPOCHS, DEVICE, target_label

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1024)
    model.cuda()
    result = defaultdict(list)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    

    epochs = EPOCHS
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

    model.zero_grad()
    for epoch in range(EPOCHS):
        train_loss = 0
        train_accuracy = 0
        valid_loss = 0
        valid_accuracy = 0

        model.train()   
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # forward
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            # loss
            train_loss_batch = outputs.loss
            train_loss += train_loss_batch.item()

            # accuracy
            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in b_labels.cpu().numpy()]
            train_accuracy += accuracy_score(true, pred)

            # backward
            train_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        ## Validation
        model.eval()
        for batch in valid_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # forward
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            # loss
            valid_loss_batch = outputs.loss
            valid_loss += valid_loss_batch.item()

            # accuracy
            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in b_labels.cpu().numpy()]
            valid_accuracy += accuracy_score(true, pred)

        ## Report status
        avg_train_loss = train_loss/len(train_dataloader)
        avg_train_accuracy = train_accuracy/len(train_dataloader)
        avg_valid_loss = valid_loss/len(valid_dataloader)
        avg_valid_accuracy = valid_accuracy/len(valid_dataloader)
        
        result['epoch'].append(epoch+1)
        result['train_loss'].append(avg_train_loss)
        result['valid_loss'].append(avg_valid_loss)
        result['train_accuracy'].append(avg_train_accuracy)
        result['valid_accuracy'].append(avg_valid_accuracy)
        log = '  | Epochs: ({}/{})  TrLs: {:.03f}  TrAcc: {:.03f}  VlLs: {:.03f}  VlAcc: {:.03f}'.format(epoch+1, EPOCHS, avg_train_loss, avg_train_accuracy, avg_valid_loss, avg_valid_accuracy)
        print('\r'+log, end='')

    print('\n  | Training complete')
    return model, result, avg_train_accuracy, avg_valid_accuracy

def model_testing(model, test_dataloader, model_info):
    global TOKENIZER

    model.eval()
    test_accuracy = 0
    preds = defaultdict(list)

    for batch in test_dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # forward
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # accuracy
        pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
        true = [label for label in b_labels.cpu().numpy()]
        test_accuracy += accuracy_score(true, pred)
        for ids, p, t in zip(b_input_ids, pred, true):
            preds['text'].append(TOKENIZER.decode(ids))
            preds['pred'].append(p)
            preds['true'].append(t)

    ## Confusion matrix
    conf_matrix = confusion_matrix(preds['true'], preds['pred'])
    fname_conf_matrix = 'conf_matrix-bert_{}.xlsx'.format(model_info)
    clauseio.save_result(result=conf_matrix, fname_result=fname_conf_matrix)

    # Report status
    avg_test_accuracy = test_accuracy/len(test_dataloader)
    return avg_test_accuracy

def show_resuluts(train_accuracy, valid_accuracy, test_accuracy, accuracies):
    global RANDOM_STATE, MAX_SENT_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE

    print('  | RANDOM_STATE  : {}'.format(RANDOM_STATE))
    print('  | MAX_SENT_LEN  : {}'.format(MAX_SENT_LEN))
    print('  | BATCH_SIZE    : {}'.format(BATCH_SIZE))
    print('  | EPOCHS        : {}'.format(EPOCHS))
    print('  | LEARNING_RATE : {}'.format(LEARNING_RATE))
    print('  |')
    print('  | Training Acc  : {}'.format(train_accuracy))
    print('  | Validation Acc: {}'.format(valid_accuracy))
    print('  | Testing Acc   : {}'.format(test_accuracy))

    accuracies['RANDOM_STATE'].append(RANDOM_STATE)
    accuracies['MAX_SENT_LEN'].append(MAX_SENT_LEN)
    accuracies['BATCH_SIZE'].append(BATCH_SIZE)
    accuracies['EPOCHS'].append(EPOCHS)
    accuracies['LEARNING_RATE'].append(LEARNING_RATE)
    accuracies['train_accuracy'].append(train_accuracy)
    accuracies['valid_accuracy'].append(valid_accuracy)
    accuracies['test_accuracy'].append(test_accuracy)
    return accuracies


if __name__ == '__main__':
    ## Filenames
    base = '1,976'
    model_version = '4.2.0'

    ## Tokenizer
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    ## Model development
    DEVICE = clausefunc.gpu_allocation()

    ## Parameters
    RANDOM_STATE = 42
    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75

    train_ratio = round(TRAIN_TEST_RATIO*TRAIN_VALID_RATIO*100)
    valid_ratio = round(TRAIN_TEST_RATIO*(1-TRAIN_VALID_RATIO)*100)
    test_ratio = round((1-TRAIN_TEST_RATIO)*100)
    
    # MAX_SENT_LEN = 128
    # BATCH_SIZE = 16
    EPOCHS = 100
    # LEARNING_RATE = 2e-4

    accuracies = defaultdict(list)
    for MAX_SENT_LEN in [64, 128]:
        for BATCH_SIZE in [8, 16, 32]:
            for LEARNING_RATE in [2e-4, 2e-5]:
                print('============================================================')
                print('BERT model development')

                ## Load corpus
                fname_data_for_modeling = 'data_for_modeling_{}.pk'.format(base)
                fpath_data_for_modeling = os.path.sep.join((clauseio.fdir_corpus, fname_data_for_modeling))
                with open(fpath_data_for_modeling, 'rb') as f:
                    data_for_modeling = pk.load(f)

                train_inputs, train_masks, train_labels = data_for_modeling['train']
                train_dataloader = clausefunc.build_dataloader(inputs=train_inputs, labels=train_labels, masks=train_masks, batch_size=BATCH_SIZE, encode='multi')
                valid_inputs, valid_masks, valid_labels = data_for_modeling['valid']
                valid_dataloader = clausefunc.build_dataloader(inputs=valid_inputs, labels=valid_labels, masks=valid_masks, batch_size=BATCH_SIZE, encode='multi')
                test_inputs, test_masks, test_labels = data_for_modeling['test']
                test_dataloader = clausefunc.build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, batch_size=BATCH_SIZE, encode='multi')

                ## Model training
                print('------------------------------------------------------------')
                print('  | Model training')

                model_info = 'V-{}_D-{}_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LR-{}'.format(model_version,
                                                                                          base, 
                                                                                          train_ratio, 
                                                                                          valid_ratio, 
                                                                                          test_ratio, 
                                                                                          BATCH_SIZE, 
                                                                                          EPOCHS, 
                                                                                          LEARNING_RATE)

                model, result, train_accuracy, valid_accuracy = model_training(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

                ## Export training result
                fname_model = 'bert_{}.pk'.format(model_info)
                fname_train_result = 'result-bert_{}_train.xlsx'.format(model_info)
                
                clauseio.save_model(model=model, fname_model=fname_model)
                clauseio.save_result(result=result, fname_result=fname_train_result)

                ## Model testing
                print('------------------------------------------------------------')
                print('  | Model testing')

                test_accuracy = model_testing(model=model, test_dataloader=test_dataloader, model_info=model_info)

                ## Export testing result
                print('------------------------------------------------------------')
                print('  | Model results')

                # fname_test_result = 'result-bert_{}_test.xlsx'.format(model_info)
                # clauseio.save_result(result=test_accuracy, fname_result=fname_test_result)
                accuracies = deepcopy(show_resuluts(train_accuracy, valid_accuracy, test_accuracy, accuracies))

    fname_accuracies = 'accuracies-bert_{}.xlsx'.format(model_version)
    clauseio.save_result(result=accuracies, fname_result=fname_accuracies)