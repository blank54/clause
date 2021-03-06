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

from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer


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




# def train_data_preparation(train):
#     global TRAIN_VALID_RATIO, RANDOM_STATE, BATCH_SIZE

#     print('============================================================')
#     print('Train data preparation')

#     train_tokenized = tokenize(data=train)
#     train_padded = padding(tokenized_docs=train_tokenized)
#     train_attention_masks = attention_masking(padded_docs=train_padded)

#     train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(train_padded, [d.labels for d in train], random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)
#     train_masks, valid_masks, _, _ = train_test_split(train_attention_masks, train_padded, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

#     return train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks

# def test_data_preparation(test):
#     global BATCH_SIZE

#     print('============================================================')
#     print('Test data preparation')

#     test_tokenized = tokenize(data=test)
#     test_padded = padding(tokenized_docs=test_tokenized)
#     test_attention_masks = attention_masking(padded_docs=test_padded)

#     return test_padded, [d.labels for d in test], test_attention_masks



def resample(inputs, labels, target_label):
    labels_encoded = clausefunc.encode_labels_binary(labels=labels, target_label=target_label)
    inputs_res, labels_res = SMOTE(random_state=RANDOM_STATE).fit_resample(inputs, labels_encoded)
    # inputs_res = inputs_res.to_numpy()
    masks_res = attention_masking(padded_docs=inputs_res)
    return inputs_res, masks_res, labels_res


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

    # train, test = data_split(corpus=corpus)
    # train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks = train_data_preparation(train=train)
    # test_inputs, test_labels, test_masks = test_data_preparation(test=test)

    ## Resampling
    print('============================================================')
    print('Data resampling: SMOTE')

    label_list = clauseio.read_label_list(version='v2')
    for target_label in label_list:
        train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(inputs, masks, labels, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)
        train_inputs, valid_inputs, train_masks, valid_masks, train_labels, valid_labels = train_test_split(train_inputs, train_masks, train_labels, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

        inputs_res, masks_res, labels_res = resample(inputs=inputs, labels=labels, target_label=target_label)
        train_inputs_res, test_inputs_res, train_masks_res, test_masks_res, train_labels_res, test_labels_res = train_test_split(inputs_res, masks_res, labels_res, random_state=RANDOM_STATE, train_size=TRAIN_TEST_RATIO)
        train_inputs_res, valid_inputs_res, train_masks_res, valid_masks_res, train_labels_res, valid_labels_res = train_test_split(train_inputs_res, train_masks_res, train_labels_res, random_state=RANDOM_STATE, train_size=TRAIN_VALID_RATIO)

        corpus_res = {}
        fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
        fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))

        corpus_res['train'] = (train_inputs, train_masks, train_labels)
        corpus_res['valid'] = (valid_inputs, valid_masks, valid_labels)
        corpus_res['test'] = (test_inputs, test_masks, test_labels)
        corpus_res['train_res'] = (train_inputs_res, train_masks_res, train_labels_res)
        corpus_res['valid_res'] = (valid_inputs_res, valid_masks_res, valid_labels_res)
        corpus_res['test_res'] = (test_inputs_res, test_masks_res, test_labels_res)

        with open(fpath_resampled, 'wb') as f:
            pk.dump(corpus_res, f)
        
        print('  | {:10}: {:5,} / {:5,} / {:5,} ({})'.format(target_label, len(train_inputs_res), len(valid_inputs_res), len(test_inputs_res), fname_resampled))