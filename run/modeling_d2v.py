#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO
clauseio = ClauseIO()

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


if __name__ == '__main__':
    ## Parameters
    parameters = {'vector_size': 100,
                  'window': 5,
                  'min_count': 5,
                  'dm': 1,
                  'negative': 5,
                  'epochs': 100,
                  'dbow_words': 1}

    ## Filenames
    fname_corpus = 'corpus_940_T-t_P-t_N-t_S-t_L-t.pk'
    fname_model = 'd2v_940_V-{}_W-{}_M-{}_E-{}.pk'.format(parameters.get('vector_size'),
                                                      parameters.get('window'),
                                                      parameters.get('min_count'),
                                                      parameters.get('epochs'))

    ## Data preparation
    print('============================================================')
    print('Input corpus: {}'.format(fname_corpus))

    corpus = clauseio.read_corpus(fname_corpus=fname_corpus)
    docs = [TaggedDocument(tags=[doc.tag], words=doc.lemmatized_text) for doc in corpus]
    
    ## Model training
    print('============================================================')
    print('Model training')

    d2v_model = Doc2Vec(documents=docs,
                        vector_size=parameters.get('vector_size'),
                        window=parameters.get('window'),
                        min_count=parameters.get('min_count'),
                        dm=parameters.get('dm'),
                        negative=parameters.get('negative'),
                        epochs=parameters.get('epochs'),
                        dbow_words=parameters.get('dbow_words'))

    ## Model verification
    test_data = 'Before delivering a traffic signal controller to site, the Contractor shall arrange a factory acceptance test in his workshop  The programmed and internally complete controller shall be connected to a labelled light board capable of simulating all traffic signal aspects controlled by that particular controller  The Contractor shall ensure that all equipment and devices are available to show that the controller fully complies with operational requirements'.split()
    
    print('============================================================')
    print('Doc2Vec verification')
    print('  | inferred vector: {} ...'.format(d2v_model.infer_vector(doc_words=test_data)[:5]))

    ## Save model
    print('============================================================')
    print('Save Doc2Vec model')
    clauseio.save_model(model=d2v_model, fname_model=fname_model)