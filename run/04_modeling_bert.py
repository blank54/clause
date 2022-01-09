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

import random
import numpy as np
import pickle as pk
from collections import defaultdict
from sklearn.metrics import accuracy_score


def model_training(train_dataloader, valid_dataloader):
    global RANDOM_STATE, EPOCHS, DEVICE, target_label

    print('------------------------------------------------------------')
    print('  | Model training')

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
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
        
        result['target_label'].append(target_label)
        result['epoch'].append(epoch+1)
        result['train_loss'].append(avg_train_loss)
        result['valid_loss'].append(avg_valid_loss)
        result['train_accuracy'].append(avg_train_accuracy)
        result['valid_accuracy'].append(avg_valid_accuracy)
        log = '  | Epochs: ({}/{})  TrLs: {:.03f}  TrAcc: {:.03f}  VlLs: {:.03f}  VlAcc: {:.03f}'.format(epoch+1, EPOCHS, avg_train_loss, avg_train_accuracy, avg_valid_loss, avg_valid_accuracy)
        print('\r'+log, end='')

    print('\n  | Training complete')
    return model, result

def model_testing(model, test_dataloader):
    print('------------------------------------------------------------')
    print('  | Model testing')

    model.eval()
    test_accuracy = 0

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

    # Report status
    avg_test_accuracy = test_accuracy/len(test_dataloader)
    print('  | Test accuracy: {:.03f}'.format(avg_test_accuracy))
    return avg_test_accuracy


if __name__ == '__main__':
    ## Filenames
    base = '1,053'

    ## Parameters
    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75

    train_ratio = round(TRAIN_TEST_RATIO*TRAIN_VALID_RATIO*100)
    valid_ratio = round(TRAIN_TEST_RATIO*(1-TRAIN_VALID_RATIO)*100)
    test_ratio = round((1-TRAIN_TEST_RATIO)*100)
    
    RESAMPLING = False
    MAX_SENT_LEN = 128
    BATCH_SIZE = 8
    RANDOM_STATE = 42

    EPOCHS = 100
    LEARNING_RATE = 2e-4

    ## Model development
    DEVICE = clausefunc.gpu_allocation()
    test_result = defaultdict(list)

    # label_list = ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'DEFINITION', 'SCOPE', 'RnR']
    label_list = ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'DEFINITION', 'SCOPE']
    # label_list = ['PAYMENT']
    for target_label in label_list:
        print('============================================================')
        print('Target category: <{}>'.format(target_label))

        ## Load corpus
        fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
        fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))
        with open(fpath_resampled, 'rb') as f:
            corpus_res = pk.load(f)

        if RESAMPLING:
            train_inputs, train_masks, train_labels = corpus_res['train_res']
            train_dataloader = clausefunc.build_dataloader(inputs=train_inputs, labels=train_labels, masks=train_mask batch_size=BATCH_SIZE,s, target_label=target_label, encode=False)
            valid_inputs, valid_masks, valid_labels = corpus_res['valid_res']
            valid_dataloader = clausefunc.build_dataloader(inputs=valid_inputs, labels=valid_labels, masks=valid_mask batch_size=BATCH_SIZE,s, target_label=target_label, encode=False)
            test_inputs, test_masks, test_labels = corpus_res['test_res']
            test_dataloader = clausefunc.build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, batch_size=BATCH_SIZE, target_label=target_label, encode=False)
        else:
            train_inputs, train_masks, train_labels = corpus_res['train']
            train_dataloader = clausefunc.build_dataloader(inputs=train_inputs, labels=train_labels, masks=train_mask batch_size=BATCH_SIZE,s, target_label=target_label, encode=True)
            valid_inputs, valid_masks, valid_labels = corpus_res['valid']
            valid_dataloader = clausefunc.build_dataloader(inputs=valid_inputs, labels=valid_labels, masks=valid_mask batch_size=BATCH_SIZE,s, target_label=target_label, encode=True)
            test_inputs, test_masks, test_labels = corpus_res['test']
            test_dataloader = clausefunc.build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, batch_size=BATCH_SIZE, target_label=target_label, encode=True)

        ## Model training
        model, result = model_training(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

        ## Export training result
        model_info = '{}_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LR-{}_RS-{}_LB-{}'.format(base, train_ratio, valid_ratio, test_ratio, BATCH_SIZE, EPOCHS, LEARNING_RATE, RESAMPLING, target_label)
        fname_model = 'result-bert3_{}.pk'.format(model_info)
        fname_train_result = 'result-bert3_{}_train.xlsx'.format(model_info)
        
        clauseio.save_model(model=model, fname_model=fname_model)
        # clauseio.save_result(result=result, fname_result=fname_train_result)

        ## Model testing
        test_accuracy = model_testing(model=model, test_dataloader=test_dataloader)
        test_result[target_label].append(test_accuracy)

    ## Export testing result
    fname_test_result = 'result-bert3_{}_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LR-{}_RS-{}_test.xlsx'.format(base, train_ratio, valid_ratio, test_ratio, BATCH_SIZE, EPOCHS, LEARNING_RATE, RESAMPLING)
    # clauseio.save_result(result=test_result, fname_result=fname_test_result)