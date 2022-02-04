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

from clauseutil import ClauseIO, ClauseFunc
clauseio = ClauseIO()
clausefunc = ClauseFunc()

import pickle as pk
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer


def tokenize(data):
    global TOKENIZER

    tokenized_docs = []
    with tqdm(total=len(data)) as pbar:
        for doc in data:
            text_as_input = '[CLS] {} [SEP]'.format(' '.join(doc.normalized_text))
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

def data_preparation(corpus):
    corpus_tokenized = tokenize(data=corpus)
    inputs = padding(tokenized_docs=corpus_tokenized)
    masks = attention_masking(padded_docs=inputs)
    labels = [d.labels for d in corpus]

    return inputs, masks, labels


if __name__ == '__main__':
    ## Parameters
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75

    MAX_SENT_LEN = 128
    BATCH_SIZE = 16
    RANDOM_STATE = 42

    ## Filenames
    base = '1,976'
    fname_corpus = 'corpus_{}_T-t_P-t_N-t_S-t_L-t.pk'.format(base)

    ## Data preparation
    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    inputs, masks, labels = data_preparation(corpus=corpus)

    train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(inputs, masks, labels, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)
    train_inputs, valid_inputs, train_masks, valid_masks, train_labels, valid_labels = train_test_split(train_inputs, train_masks, train_labels, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

    data_for_modeling = {}
    fname_data_for_modeling = 'data_for_modeling_{}.pk'.format(base)
    fpathampled = os.path.sep.join((clauseio.fdir_corpus, fname_data_for_modeling))

    data_for_modeling['train'] = (train_inputs, train_masks, train_labels)
    data_for_modeling['valid'] = (valid_inputs, valid_masks, valid_labels)
    data_for_modeling['test'] = (test_inputs, test_masks, test_labels)

    with open(fpathampled, 'wb') as f:
        pk.dump(data_for_modeling, f)
    
    print('{:5,} / {:5,} / {:5,} ({})'.format(len(train_inputs), len(valid_inputs), len(test_inputs), fname_data_for_modeling))