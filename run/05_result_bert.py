#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from clauseutil import ClauseIO, ClauseFunc
clauseio = ClauseIO()
clausefunc = ClauseFunc()

import pickle as pk
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import torch
from transformers import BertTokenizer


if __name__ == '__main__':
    
    ## Parameters
    TRAIN_TEST_RATIO = 0.8
    TRAIN_VALID_RATIO = 0.75

    train_ratio = round(TRAIN_TEST_RATIO*TRAIN_VALID_RATIO*100)
    valid_ratio = round(TRAIN_TEST_RATIO*(1-TRAIN_VALID_RATIO)*100)
    test_ratio = round((1-TRAIN_TEST_RATIO)*100)
    
    RESAMPLING = False
    BATCH_SIZE = 8

    EPOCHS = 100
    LEARNING_RATE = 2e-4

    ## Tokenizer
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    ## GPU allocation
    DEVICE = clausefunc.gpu_allocation()


    ## Filenames
    base = '1,053'
    version = '3.1'
    model_info = 'V-{}_D-{}_TR-{}_VL-{}_TS-{}_BS-{}_EP-{}_LR-{}_RS-{}'.format(version,
                                                                              base, 
                                                                              train_ratio, 
                                                                              valid_ratio, 
                                                                              test_ratio, 
                                                                              BATCH_SIZE, 
                                                                              EPOCHS, 
                                                                              LEARNING_RATE, 
                                                                              RESAMPLING)
    ## Prediction
    performances = defaultdict(list)

    label_list = ['PAYMENT', 'TEMPORAL', 'METHOD', 'QUALITY', 'SAFETY', 'DEFINITION', 'SCOPE', 'RnR']
    for target_label in label_list:
        print('============================================================')
        print('Target category: <{}>'.format(target_label))

        ## Load corpus
        fname_resampled = 'corpus_res_{}_{}.pk'.format(base, target_label)
        fpath_resampled = os.path.sep.join((clauseio.fdir_corpus, fname_resampled))
        with open(fpath_resampled, 'rb') as f:
            corpus_res = pk.load(f)

        if RESAMPLING:
            test_inputs, test_masks, test_labels = corpus_res['test_res']
            test_dataloader = clausefunc.build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, batch_size=BATCH_SIZE, target_label=target_label, encode=False)
        else:
            test_inputs, test_masks, test_labels = corpus_res['test']
            test_dataloader = clausefunc.build_dataloader(inputs=test_inputs, labels=test_labels, masks=test_masks, batch_size=BATCH_SIZE, target_label=target_label, encode=True)

        ## Load model
        fname_model = 'bert_{}_LB-{}.pk'.format(model_info, target_label)
        model = clauseio.read_model(fname_model=fname_model)

        ## Model predict
        preds = defaultdict(list)

        model.eval()
        with tqdm(total=len(test_dataloader)) as pbar:
            for batch in test_dataloader:
                pbar.update(1)

                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # forward
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

                # predict
                pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
                true = [label for label in b_labels.cpu().numpy()]

                # record
                for ids, p, t in zip(b_input_ids, pred, true):
                    preds['text'].append(TOKENIZER.decode(ids))
                    preds['pred'].append(p)
                    preds['true'].append(t)

        ## Save prediction
        fname_pred_result = 'pred-bert_{}_LB-{}.xlsx'.format(model_info, target_label)
        clauseio.save_result(result=preds, fname_result=fname_pred_result)

        ## Confusion matrix
        conf_matrix = confusion_matrix(preds['true'], preds['pred'])
        fname_conf_matrix = 'conf_matrix-bert_{}_LB-{}.xlsx'.format(model_info, target_label)
        clauseio.save_result(result=conf_matrix, fname_result=fname_conf_matrix)

        ## Model performance
        precision_pos = precision_score(preds['true'], preds['pred'], pos_label=1)
        recall_pos = recall_score(preds['true'], preds['pred'], pos_label=1)
        f1_score_pos = f1_score(preds['true'], preds['pred'], pos_label=1)

        precision_neg = precision_score(preds['true'], preds['pred'], pos_label=0)
        recall_neg = recall_score(preds['true'], preds['pred'], pos_label=0)
        f1_score_neg = f1_score(preds['true'], preds['pred'], pos_label=0)

        performances['label'].append(target_label)
        performances['precision_pos'].append(precision_pos)
        performances['recall_pos'].append(recall_pos)
        performances['f1_score_pos'].append(f1_score_pos)
        performances['precision_neg'].append(precision_neg)
        performances['recall_neg'].append(recall_neg)
        performances['f1_score_neg'].append(f1_score_neg)

    ## Save performance
    fname_performances = 'performances-bert_{}.xlsx'.format(model_info)
    clauseio.save_result(result=performances, fname_result=fname_performances)