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

from object import Doc
from clauseutil import ClausePath, ClauseFunc, ClauseEval
clausepath = ClausePath()
clausefunc = ClauseFunc()
clauseeval = ClauseEval()

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from collections import defaultdict


def data_split(corpus):
    global TRAIN_TEST_RATIO, RANDOM_STATE

    print('============================================================')
    print('Train-test split')
    print('  | Train-Test ratio: {}'.format(TRAIN_TEST_RATIO))

    train, test = train_test_split(corpus, train_size=TRAIN_TEST_RATIO, random_state=RANDOM_STATE)

    print('  | # of corpus: {:,}'.format(len(corpus)))
    print('  | # of train : {:,}'.format(len(train)))
    print('  | # of test  : {:,}'.format(len(test)))

    return train, test

def tokenize(data):
    global TOKENIZER

    tokenized_docs = []
    with tqdm(total=len(data)) as pbar:
        for doc in data:
            text_as_input = '[CLS] {} [SEP]'.format(doc.normalized_text)
            tokenized_docs.append(TOKENIZER.tokenize(text_as_input))
            pbar.update(1)

    return tokenized_docs

def padding(tokenized_docs):
    global TOKENIZER, MAX_SENT_LEN

    docs_by_ids = []
    with tqdm(total=len(tokenized_docs)) as pbar:
        for doc in tokenized_docs:
            docs_by_ids.append(TOKENIZER.convert_tokens_to_ids(doc))
            pbar.update(1)

    padded_docs = pad_sequences(docs_by_ids, maxlen=MAX_SENT_LEN, dtype='long', truncating='post', padding='post')

    return padded_docs

def attention_masking(padded_docs):
    attention_masks = []
    for seq in padded_docs:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    return attention_masks

def build_dataloader(inputs, labels, masks, target_label, option):
    inputs = torch.tensor(inputs)
    labels = torch.tensor(clausefunc.encode_labels_binary(labels=labels, target_label=target_label))
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)

    if option == 'train':
        sampler = RandomSampler(data)
    elif option == 'valid':
        sampler = SequentialSampler(data)
    elif option == 'test':
        sampler = RandomSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

def train_data_preparation(train):
    global TRAIN_VALID_RATIO, RANDOM_STATE, BATCH_SIZE

    print('============================================================')
    print('Train data preparation')

    train_tokenized = tokenize(data=train)
    train_padded = padding(tokenized_docs=train_tokenized)
    train_attention_masks = attention_masking(padded_docs=train_padded)

    train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(train_padded, [d.labels for d in train], random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)
    train_masks, valid_masks, _, _ = train_test_split(train_attention_masks, train_padded, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

    return train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks


def test_data_preparation(test):
    global BATCH_SIZE

    print('============================================================')
    print('Test data preparation')

    test_tokenized = tokenize(data=test)
    test_padded = padding(tokenized_docs=test_tokenized)
    test_attention_masks = attention_masking(padded_docs=test_padded)

    return test_padded, [d.labels for d in test], test_attention_masks

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

def model_training(train_dataloader, valid_dataloader, result):
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

    model.zero_grad()
    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        valid_accuracy = 0
        nb_valid_steps = 0

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

        ## Report status
        avg_train_loss = train_loss/len(train_dataloader)
        avg_valid_loss = valid_loss/len(valid_dataloader)
        current_accuracy = valid_accuracy/nb_valid_steps
        
        result['target_label'].append(target_label)
        result['epoch'].append(epoch+1)
        result['train_loss'].append(avg_train_loss)
        result['valid_loss'].append(avg_valid_loss)
        result['valid_accuracy'].append(current_accuracy)
        log = '  | Epochs: ({}/{})   TrainLoss: {:.03f}   ValidLoss: {:.03f}   ValidAccuracy: {:.03f}'.format(epoch+1, EPOCHS, avg_train_loss, avg_valid_loss, current_accuracy)
        print('\r'+log, end='')

    print('\n  | Training complete')
    return model, result

def model_evaluation(model, test_dataloader):
    global DEVICE

    print('------------------------------------------------------------')
    print('  | Model evaluation')

    test_accuracy = 0
    nb_test_steps = 0

    model.eval()
    for step, batch in enumerate(test_dataloader):
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
    total_accuracy = test_accuracy/nb_test_steps
    print('  | TestAccuracy: {:.03f}'.format(total_accuracy))
    print('  | Evaluation complete')


if __name__ == '__main__':
    ## Parameters
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75
    
    MAX_SENT_LEN = 128
    BATCH_SIZE = 16
    RANDOM_STATE = 42

    EPOCHS = 30
    LEARNING_RATE = 2e-5

    ## Filenames
    fname_corpus = 'corpus_T-t_P-t_N-t_S-t_L-t.pk'

    train_ratio = round(TRAIN_TEST_RATIO*TRAIN_VALID_RATIO*100)
    valid_ratio = round(TRAIN_TEST_RATIO*(1-TRAIN_VALID_RATIO)*100)
    test_ratio = round((1-TRAIN_TEST_RATIO)*100)
    
    ## Data preparation
    corpus = clausefunc.read_corpus(fname_corpus=fname_corpus)
    train, test = data_split(corpus=corpus)
    
    train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks = train_data_preparation(train=train)
    test_inputs, test_labels, test_masks = test_data_preparation(test=test)

    ## Model development
    DEVICE = gpu_allocation()
    for target_label in ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'RnR', 'DEFINITION', 'SCOPE']:
        print('============================================================')
        print('Target category: <{}>'.format(target_label))

        ## Build dataloader
        train_dataloader = build_dataloader(inputs=train_inputs, labels=train_labels, masks=train_masks, target_label=target_label, option='train')
        valid_dataloader = build_dataloader(inputs=valid_inputs, labels=valid_labels, masks=valid_masks, target_label=target_label, option='valid')
        test_dataloader = build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, target_label=target_label, option='test')

        ## Model training
        result = defaultdict(list)
        model, result = model_training(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, result=result)

        ## Model evaluation
        model_evaluation(model=model, test_dataloader=test_dataloader)

        ## Export result
        fname_result = 'result_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LB-{}.xlsx'.format(train_ratio, valid_ratio, test_ratio, BATCH_SIZE, EPOCHS, target_label)
        clausefunc.save_result(result=result, fname_result=fname_result)