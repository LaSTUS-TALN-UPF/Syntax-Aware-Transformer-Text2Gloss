# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:46:31 2021

@author: SNT
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "./utils/")

from rouge import rouge
from sacreBLEU_script import compute_sacre_bleu

from nltk.translate.meteor_score import meteor_score
from pyter import ter

REF_FILENAME = 'data/glosses_test.txt'
WORKING_DIR = 'T2G_syntax' # 'T2G_no_syntax'
EVAL_EPOCHS = range(5,501,5)


#%%############################################################################
'''                         ACCOMODATING ES-EN DATA                         '''
###############################################################################

with open(REF_FILENAME, 'rb') as f:
    ref_data = f.read().decode().split('\n')
    
columns = ["rouge_1/f_score", "rouge_1/r_score", "rouge_1/p_score",
           "rouge_2/f_score", "rouge_2/r_score", "rouge_2/p_score",
           "ROUGE-L (F1-score)", "rouge_l/r_score", "rouge_l/p_score",
           'Sacrebleu', 'TER', 'METEOR']

#%%############################################################################
'''                            COMPUTING METRICS                            '''
###############################################################################    
results_top_container = np.zeros(shape = (len(EVAL_EPOCHS), len(columns)))
imodel = 0
for it in EVAL_EPOCHS:

    with open(WORKING_DIR+'/generated_text/'+'{}.txt'.format(it), 'rb') as f:
        generated_text = f.read().decode()
        generated_text = [s.strip() for s in generated_text.split('\n')]
    
    hyp = [s for s in generated_text]
    ref = [s for s in ref_data]
    rouge_scores = rouge(hyp, ref)
    
    results_top_container[imodel, 0] = rouge_scores["rouge_1/f_score"]
    results_top_container[imodel, 1] = rouge_scores["rouge_1/r_score"]
    results_top_container[imodel, 2] = rouge_scores["rouge_1/p_score"]
    results_top_container[imodel, 3] = rouge_scores["rouge_2/f_score"]
    results_top_container[imodel, 4] = rouge_scores["rouge_2/r_score"]
    results_top_container[imodel, 5] = rouge_scores["rouge_2/p_score"]        
    results_top_container[imodel, 6] = rouge_scores["rouge_l/f_score"]        
    results_top_container[imodel, 7] = rouge_scores["rouge_l/r_score"]        
    results_top_container[imodel, 8] = rouge_scores["rouge_l/p_score"]   

    try:
        results_top_container[imodel, 9] = compute_sacre_bleu(hyp, ref, tokenize = 'char')
    except EOFError:
        results_top_container[imodel, 9] = 0
    
    results_top_container[imodel, 10] = np.mean([ter(h.split(),r.split()) for h,r in zip(hyp,ref)])   
    
    ref = [[s] for s in ref_data]
    results_top_container[imodel, 11] =  np.mean([meteor_score(r, h) for r, h in zip(ref, hyp)])
    imodel += 1
    
results = pd.DataFrame(results_top_container, columns = columns, index =EVAL_EPOCHS)
dataset_type = REF_FILENAME.split('_')[-1].split('.')[0]
results.to_excel('{}/metrics_{}.xlsx'.format(WORKING_DIR, dataset_type))

