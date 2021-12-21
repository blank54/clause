#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
http://yonghee.io/bert_binary_classification_naver/
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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import random
import numpy as np
import pickle as pk
from collections import defaultdict


def build_dataloader(inputs, labels, masks, target_label, option, encode):
    inputs = torch.tensor(inputs)    
    masks = torch.tensor(masks)
    
    if encode:
        labels = torch.tensor(clausefunc.encode_labels_binary(labels=labels, target_label=target_label))
    else:
        labels = torch.tensor(labels)

    data = TensorDataset(inputs, masks, labels)

    if option == 'train':
        sampler = RandomSampler(data)
    elif option == 'valid':
        sampler = SequentialSampler(data)
    elif option == 'test':
        sampler = RandomSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

def gpu_allocation():
    print('============================================================')
    print('GPU allocation')

    if torch.cuda.is_available():    
        device = torch.device('cuda')
        print('  | There are {} GPU(s) available.'.format(torch.cuda.device_count()))
        print('  | Current CUDA device: {}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        print('  | No GPU available, using the CPU instead.')

    return device

def model_training(train_dataloader, valid_dataloader, test_dataloader):
    global RANDOM_STATE, EPOCHS, DEVICE, target_label

    print('------------------------------------------------------------')
    print('  | Model training')

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE, #학습률
                      eps=1e-8 #0으로 나누는 것을 방지하기 위한 epsilon 값
                      )

    epochs = EPOCHS
    total_steps = len(train_dataloader) * epochs #총 훈련 스텝
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

    result = defaultdict(list)
    model.zero_grad()
    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0

        valid_accuracy = 0
        nb_valid_steps = 0

        test_accuracy = 0
        nb_test_steps = 0

        model.train()   
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss = outputs[0] # 로스 구함
            train_loss += loss.item() # 총 로스 계산

            loss.backward() # Backward 수행으로 그래디언트 계산
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
            optimizer.step() # 그래디언트를 통해 가중치 파라미터 업데이트
            scheduler.step() # 스케줄러로 학습률 감소
            model.zero_grad() # 그래디언트 초기화

        ## Validation
        model.eval()
        for batch in valid_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
            loss = outputs[0]
            valid_loss += loss.item()

            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_valid_accuracy = clauseeval.flat_accuracy(logits, label_ids)
            valid_accuracy += tmp_valid_accuracy
            nb_valid_steps += 1

        ## Test
        model.eval()
        for batch in test_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_test_accuracy = clauseeval.flat_accuracy(logits, label_ids)
            test_accuracy += tmp_test_accuracy
            nb_test_steps += 1

        ## Report result
        test_accuracy = test_accuracy/nb_test_steps


        ## Report status
        avg_train_loss = train_loss/len(train_dataloader)
        avg_valid_loss = valid_loss/len(valid_dataloader)
        valid_accuracy = valid_accuracy/nb_valid_steps
        
        result['target_label'].append(target_label)
        result['epoch'].append(epoch+1)
        result['train_loss'].append(avg_train_loss)
        result['valid_loss'].append(avg_valid_loss)
        result['valid_accuracy'].append(valid_accuracy)
        result['test_accuracy'].append(test_accuracy)
        log = '  | Epochs: ({}/{})  TrLs: {:.03f}  VlLs: {:.03f}  VlAcc: {:.03f}  TsAcc: {:.03f}'.format(epoch+1, EPOCHS, avg_train_loss, avg_valid_loss, valid_accuracy, test_accuracy)
        print('\r'+log, end='')

    print('\n  | Training complete')
    return model, result


if __name__ == '__main__':
    ## Parameters
    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75
    
    MAX_SENT_LEN = 128
    BATCH_SIZE = 16
    RANDOM_STATE = 42

    EPOCHS = 1000
    LEARNING_RATE = 2e-4

    ## Filenames
    base = '1,053'

    # train_ratio = round(TRAIN_TEST_RATIO*TRAIN_VALID_RATIO*100)
    # valid_ratio = round(TRAIN_TEST_RATIO*(1-TRAIN_VALID_RATIO)*100)
    # test_ratio = round((1-TRAIN_TEST_RATIO)*100)
    
    ## Model development
    DEVICE = gpu_allocation()
    label_list = ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'RnR', 'DEFINITION', 'SCOPE']
    for target_label in label_list:
        print('============================================================')
        print('Target category: <{}>'.format(target_label))

        ## Load corpus
        fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
        fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))
        with open(fpath_resampled, 'rb') as f:
            corpus_res = pk.load(f)

        train_inputs_res, train_masks_res, train_labels_res = corpus_res['train']
        valid_inputs, valid_masks, valid_labels = corpus_res['valid']
        test_inputs, test_masks, test_labels = corpus_res['test']

        ## Build dataloader
        train_dataloader = build_dataloader(inputs=train_inputs_res, labels=train_labels_res, masks=train_masks_res, target_label=target_label, option='train', encode=False)
        valid_dataloader = build_dataloader(inputs=valid_inputs, labels=valid_labels, masks=valid_masks, target_label=target_label, option='valid', encode=True)
        test_dataloader = build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, target_label=target_label, option='test', encode=True)

        ## Model training
        model, result = model_training(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)

        ## Export result
        fname_result = 'result_{}-hotfix_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LR-{}_LB-{}.xlsx'.format(base, train_ratio, valid_ratio, test_ratio, BATCH_SIZE, EPOCHS, LEARNING_RATE, target_label)
        clauseio.save_result(result=result, fname_result=fname_result)