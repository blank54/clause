#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from provutil import ProvFunc
provfunc = ProvFunc()

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


if __name__ == '__main__':
    ## Parameters


    ## Filenames
    fname_corpus = 'provision_labeled.pk'

    ## Data preparation
    corpus = provfunc.read_corpus(fname_corpus=fname_corpus)
    docs_preprocessed = #TODO: PoS tagging, stopword removal, lemmatization
    docs_for_d2v = [TaggedDocument(words=doc.normalized_text.split(), tags=doc.tag) for doc in docs_preprocessed]
    print(docs_for_d2v[0])


    ## Model development
