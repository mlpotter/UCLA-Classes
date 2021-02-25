# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:00:11 2021

@author: Michael Potter
"""

import numpy as np
from scipy.stats import ttest_ind
from utils import *
from models import GCN
from visualize import *
import torch
import pandas as pd

GCN_accuracy = pd.read_excel("mlp_results/accuracy_for_each_corpus.xlsx",sheet_name='GCN')
Logistic_accuracy = pd.read_excel("mlp_results/accuracy_for_each_corpus.xlsx",sheet_name='Logistic')

device = torch.device('cuda')

for corpus in GCN_accuracy.columns:

    data = load_pytorch_geometric_data(corpus)
    data = data.to(device)
    
    rv1 = GCN_accuracy[corpus]
    rv2 = Logistic_accuracy[corpus]
    rv1_mean = np.mean(rv1); rv2_mean = np.mean(rv2);
    rv1_std = np.std(rv1); rv2_std = np.std(rv2);
    
    print("-"*25,corpus,"-"*25)
    print("GCN Accuracy: {:.5f}+-{:.5f}".format(rv1_mean,rv1_std))
    print("Logistic Accuracy: {:.5f}+-{:.5f}".format(rv2_mean,rv2_std))
    if np.mean(rv1) > np.mean(rv2):
        _,p_value = ttest_ind(rv1,rv2,equal_var=False,alternative='greater')
        print("The difference is statistically significant: {}".format(p_value< 0.05))
    
    best_model = torch.load("model_best_"+corpus+".pt").cuda()
    
    t_sne(best_model,data,corpus=corpus)
    
    torch.cuda.empty_cache()
   
   