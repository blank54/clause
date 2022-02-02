#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Doc
from clauseutil import ClausePath, ClauseIO, ClauseFunc
clausepath = ClausePath()
clauseio = ClauseIO()
clausefunc = ClauseFunc()

import re
from tqdm import tqdm
from copy import deepcopy

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def tokenize(input):
    return [w.strip() for w in input.strip().split(' ')]

def pos_tagging(input):
    return ['{}//{}'.format(token, pos) for token, pos in nltk.pos_tag(input)]

def normalize(input, do_lower=True, do_marking=True):
    global do_pos_tagging

    if do_pos_tagging:
        sent = [token.split('//')[0] for token in input]
    else:
        sent = deepcopy(input)

    if do_lower:
        sent = deepcopy([w.lower() for w in sent])
    else:
        pass

    if do_marking:
        for i, w in enumerate(sent):
            if re.match('www.', str(w)):
                sent[i] = 'URL'
            elif re.search('\d+\d\.\d+', str(w)):
                sent[i] = 'REF'
            elif re.match('\d', str(w)):
                sent[i] = 'NUM'
            else:
                continue
    else:
        pass

    text = ' '.join(sent)
    text = text.replace('?', '')
    text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)

    text = text.replace(' / ', '/')
    text = re.sub('\.+\.', ' ', text)
    text = text.replace('\\\\', '\\').replace('\\r\\n', '')
    text = text.replace('\n', '  ')
    text = re.sub('\. ', '  ', text)
    text = re.sub('\s+\s', ' SENTSEP ', text).strip()
    output = text.split(' ')

    return output

def remove_stopword(input):
    global do_pos_tagging

    fname_stopwords = 'stopwords.txt'
    fpath_stopwords = os.path.sep.join((clausepath.fdir_thesaurus, fname_stopwords))
    with open(fpath_stopwords, 'r', encoding='utf-8') as f:
        stopwords = [stopword.strip() for stopword in f.read().strip().split('\n')]

    if do_pos_tagging:
        return [w for w in input if w.split('//')[0] not in stopwords]
    else:
        return [w for w in input if w not in stopwords]

def lemmatize(input):
    global do_pos_tagging

    if do_pos_tagging:
        return [lemmatizer.lemmatize(w.split('//')[0]) for w in input]
    else:
        return [lemmatizer.lemmatize(w) for w in input]

def preprocess(corpus, do_tokenization, do_pos_tagging, do_normalization, do_stopword_removal, do_lemmatization):
    corpus_preprocessed = []

    with tqdm(total=len(corpus)) as pbar:
        for doc in corpus:
            input_text = deepcopy(doc.text)

            if do_tokenization:
                input_text = deepcopy(tokenize(input=doc.text))
                tokenized_text = deepcopy(input_text)
            else:
                tokenized_text = ''
                pass

            if do_pos_tagging:
                input_text = deepcopy(pos_tagging(input=input_text))
                pos_tagged_text = deepcopy(input_text)
            else:
                pos_tagged_text = ''
                pass

            if do_normalization:
                input_text = deepcopy(normalize(input=input_text))
                normalized_text = deepcopy(input_text)
            else:
                normalized_text = ''
                pass

            if do_stopword_removal:
                input_text = deepcopy(remove_stopword(input=input_text))
                stopword_removed_text = deepcopy(input_text)
            else:
                stopword_removed_text = ''
                pass

            if do_lemmatization:
                input_text = deepcopy(lemmatize(input=input_text))
                lemmatized_text = deepcopy(input_text)
            else:
                lemmatized_text = ''
                pass

            doc_preprocessed = deepcopy(doc)
            doc_preprocessed.tokenized_text = tokenized_text
            doc_preprocessed.pos_tagged_text = pos_tagged_text
            doc_preprocessed.normalized_text = normalized_text
            doc_preprocessed.stopword_removed_text = stopword_removed_text
            doc_preprocessed.lemmatized_text = lemmatized_text

            corpus_preprocessed.append(doc_preprocessed)
            pbar.update(1)

    return corpus_preprocessed

def verify_preprocess(corpus):
    for doc in corpus[:3]:
        print('--------------------------------------------------')
        print('  | Tag             : {}'.format(doc.tag))
        print('  | Text            : {}...'.format(doc.text[:50]))
        print('  | Labels          : {}'.format(', '.join(doc.labels)))
        print('  | Tokenized       : {}...'.format(', '.join(doc.tokenized_text[:5])))
        print('  | PoS tagged      : {}...'.format(', '.join(doc.pos_tagged_text[:5])))
        print('  | Normalized      : {}...'.format(', '.join(doc.normalized_text[:5])))
        print('  | Stopword removed: {}...'.format(', '.join(doc.stopword_removed_text[:5])))
        print('  | Lemmatized      : {}...'.format(', '.join(doc.lemmatized_text[:5])))


if __name__ == '__main__':
    ## Parameters
    try:
        argv_tokenization = sys.argv[1]
    except IndexError:
        argv_tokenization = 't'
    try:
        argv_pos_tagging = sys.argv[2]
    except IndexError:
        argv_pos_tagging = 't'
    try:
        argv_normalization = sys.argv[3]
    except IndexError:
        argv_normalization = 't'
    try:
        argv_stopword_removal = sys.argv[4]
    except IndexError:
        argv_stopword_removal = 't'
    try:
        argv_lemmatization = sys.argv[5]
    except IndexError:
        argv_lemmatization = 't'

    do_tokenization = clauseio.argv2bool(argv_tokenization)
    do_pos_tagging = clauseio.argv2bool(argv_pos_tagging)
    do_normalization = clauseio.argv2bool(argv_normalization)
    do_stopword_removal = clauseio.argv2bool(argv_stopword_removal)
    do_lemmatization = clauseio.argv2bool(argv_lemmatization)

    ## Filenames
    base = '1,976'
    fname_corpus = 'corpus_{}.pk'.format(base)
    fname_corpus_preprocessed = 'corpus_{}_T-{}_P-{}_N-{}_S-{}_L-{}.pk'.format(base,
                                                                               argv_tokenization,
                                                                               argv_pos_tagging,
                                                                               argv_normalization,
                                                                               argv_stopword_removal,
                                                                               argv_lemmatization)

    ## Data preparation
    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)

    ## Preprocess
    corpus_preprocessed = preprocess(corpus=corpus,
                                     do_tokenization=do_tokenization,
                                     do_pos_tagging=do_pos_tagging,
                                     do_normalization=do_normalization,
                                     do_stopword_removal=do_stopword_removal,
                                     do_lemmatization=do_lemmatization)

    ## Verification
    print('============================================================')
    print('Parameters')
    print('  | do_tokenization    : {}'.format(do_tokenization))
    print('  | do_pos_tagging     : {}'.format(do_pos_tagging))
    print('  | do_normalization   : {}'.format(do_normalization))
    print('  | do_stopword_removal: {}'.format(do_stopword_removal))
    print('  | do_lemmatization   : {}'.format(do_lemmatization))

    print('============================================================')
    print('Verify preprocess')
    verify_preprocess(corpus=corpus_preprocessed)

    ## Save corpus
    print('============================================================')
    print('Save corpus')
    clauseio.save_corpus(corpus=corpus_preprocessed, fname_corpus=fname_corpus_preprocessed)