#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from provutil import ProvPath, ProvFunc, ProvEval
provpath = ProvPath()
provfunc = ProvFunc()
proveval = ProvEval()

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

def select_label(corpus, target_label):
    print('============================================================')
    print('Target label: {}'.format(target_label))

    corpus_targetted = []
    for doc in corpus:
        if target_label in doc.labels:
            doc.label = 1
        else:
            doc.label = 0
        corpus_targetted.append(doc)

    print('  | # of target docs   : {:,}'.format(len([d for d in corpus_targetted if d.label == 1])))
    print('  | # of remaining docs: {:,}'.format(len([d for d in corpus_targetted if d.label == 0])))

    return corpus_targetted

def data_split(corpus):
    global TRAIN_TEST_RATIO

    print('============================================================')
    print('Train-test split')
    print('  | Train-Test ratio: {}'.format(TRAIN_TEST_RATIO))

    train, test = train_test_split(corpus, train_size=TRAIN_TEST_RATIO)

    print('  | # of corpus: {:,}'.format(len(corpus)))
    print('  | # of train : {:,}'.format(len(train)))
    print('  | # of test  : {:,}'.format(len(test)))

    return train, test

def tokenize(data):
    global TOKENIZER

    print('============================================================')
    print('Tokenization')


    tokenized_docs = []
    with tqdm(total=len(data)) as pbar:
        for doc in data:
            tokenized_docs.append(TOKENIZER.tokenize(doc.text_as_input))
            pbar.update(1)

    return tokenized_docs

def padding(tokenized_docs):
    global TOKENIZER, MAX_SENT_LEN

    print('============================================================')
    print('Padding')

    docs_by_ids = []
    with tqdm(total=len(tokenized_docs)) as pbar:
        for doc in tokenized_docs:
            docs_by_ids.append(TOKENIZER.convert_tokens_to_ids(doc))
            pbar.update(1)

    padded_docs = pad_sequences(docs_by_ids, maxlen=MAX_SENT_LEN, dtype='long', truncating='post', padding='post')

    return padded_docs

def attention_masking(padded_docs):
    print('============================================================')
    print('Attention masking')

    attention_masks = []
    for seq in padded_docs:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    return attention_masks

def data_adjustment(inputs, labels, masks, option):
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)

    if option == 'train':
        sampler = RandomSampler(data)
    elif option == 'valid':
        sampler = SequentialSampler(data)
    elif option == 'test':
        pass

    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
    
    return dataloader

def train_data_preparation(train_padded, train_attention_masks):
    global TRAIN_VALID_RATIO, RANDOM_STATE, BATCH_SIZE

    print('============================================================')
    print('Train data preparation')

    train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(train_padded, [d.label for d in train], random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)
    train_masks, valid_masks, _, _ = train_test_split(train_attention_masks, train_padded, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

    train_dataloader = data_adjustment(inputs=train_inputs, labels=train_labels, masks=train_masks, option='train')
    valid_dataloader = data_adjustment(inputs=valid_inputs, labels=valid_labels, masks=valid_masks, option='valid')

    return train_dataloader, valid_dataloader

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

def model_training(train_dataloader, valid_dataloader, device):
    global RANDOM_STATE, EPOCHS

    print('============================================================')
    print('Model identification')

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.cuda()

    print('============================================================')
    print('Setup schedular')

    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE, #학습률
                      eps=1e-8 #0으로 나누는 것을 방지하기 위한 epsilon 값
                      )

    epochs = EPOCHS
    total_steps = len(train_dataloader) * epochs #총 훈련 스텝
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    print('============================================================')
    print('Model training')

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

    model.zero_grad()
    for epoch in range(EPOCHS):
        ## Training
        total_loss = 0 # 로스 초기화
        model.train() # 훈련모드로 변경
            
        for step, batch in enumerate(train_dataloader): # 데이터로더에서 배치만큼 반복하여 가져옴
            if step % 500 == 0 and not step == 0: # 경과 정보 표시
                elapsed = proveval.format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            batch = tuple(t.to(device) for t in batch) # 배치를 GPU에 넣음
            b_input_ids, b_input_mask, b_labels = batch # 배치에서 데이터 추출

            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
            
            loss = outputs[0] # 로스 구함
            total_loss += loss.item() # 총 로스 계산
            loss.backward() # Backward 수행으로 그래디언트 계산
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
            optimizer.step() # 그래디언트를 통해 가중치 파라미터 업데이트

            scheduler.step() # 스케줄러로 학습률 감소
            model.zero_grad() # 그래디언트 초기화

        ## Validation
        model.eval() # 평가모드로 변경

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in valid_dataloader: # 데이터로더에서 배치만큼 반복하여 가져옴
            batch = tuple(t.to(device) for t in batch) # 배치를 GPU에 넣음
            b_input_ids, b_input_mask, b_labels = batch # 배치에서 데이터 추출
            
            with torch.no_grad(): # 그래디언트 계산 안함
                # Forward 수행
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            logits = outputs[0] # 로스 구함

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = proveval.flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        ## Report status
        avg_train_loss = total_loss / len(train_dataloader) # 평균 로스 계산
        current_accuracy = eval_accuracy/nb_eval_steps
        print('  | ({}/{}) Loss: {:.02f} | Accuracy: {:.02f}'.format(epoch+1, EPOCHS, avg_train_loss, current_accuracy))

    print('============================================================')
    print('Training complete')

    return model




if __name__ == '__main__':
    ## Parameters
    TARGET_LABEL = 'METHOD'

    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    TRAIN_TEST_RATIO = 0.7
    TRAIN_VALID_RATIO = 0.9
    
    MAX_SENT_LEN = 128
    BATCH_SIZE = 16
    RANDOM_STATE = 42

    EPOCHS = 100
    LEARNING_RATE = 2e-5

    ## FNAME
    fname_corpus = 'provision_labeled.pk'

    ## Data import
    corpus = provfunc.read_corpus(fname_corpus=fname_corpus)
    corpus_targetted = select_label(corpus=corpus, target_label=TARGET_LABEL)
    train, test = data_split(corpus=corpus_targetted)

    ## Trainset
    train_tokenized = tokenize(data=train)
    train_padded = padding(tokenized_docs=train_tokenized)
    train_attention_masks = attention_masking(padded_docs=train_padded)
    train_dataloader, valid_dataloader = train_data_preparation(train_padded=train_padded, train_attention_masks=train_attention_masks)

    ## Model identification
    device = gpu_allocation()
    model = model_training(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, device=device)