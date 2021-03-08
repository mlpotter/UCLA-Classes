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
import argparse 
import sys

parser = argparse.ArgumentParser()


parser.add_argument('--t_sne',dest='t_sne',help='True to perform T-SNE visualizatino with GCN',action='store_true')
parser.add_argument('--ttest',dest='ttest',help='True to perform two sample welch t-test between GCN and other model',action='store_true')
parser.add_argument('--model_comparison', type=str, help='Other model to compare against GCN in excel spread sheet', default='Logistic')
parser.add_argument('--flip',dest='flip',help='Whether to flip comparison or not',action='store_true')

args = parser.parse_args()


ttest = args.ttest
t_sne = args.t_sne
model_comparison = args.model_comparison

if model_comparison not in ["Logistic","BERT","GCN_Add","GCN_Dropout"]:
  sys.exit("Do not have results for {} model".format(model_comparison))

if ttest:
  GCN_accuracy = pd.read_excel("mlp_results/accuracy_for_each_corpus.xlsx",sheet_name='GCN')
  Logistic_accuracy = pd.read_excel("mlp_results/accuracy_for_each_corpus.xlsx",sheet_name=model_comparison)
  
  
  for corpus in GCN_accuracy.columns:
  
      
      rv1 = GCN_accuracy[corpus]
      rv2 = Logistic_accuracy[corpus]
      rv1_mean = np.mean(rv1); rv2_mean = np.mean(rv2);
      rv1_std = np.std(rv1); rv2_std = np.std(rv2);
      
      print("-"*25,corpus,"-"*25)
      print("GCN Accuracy: {:.5f}+-{:.5f}".format(rv1_mean,rv1_std))
      print("{} Accuracy: {:.5f}+-{:.5f}".format(model_comparison,rv2_mean,rv2_std))
      
      if np.mean(rv1) > np.mean(rv2):
          _,p_value = ttest_ind(rv1,rv2,equal_var=False,alternative='greater')
          print("The GCN > {} difference is statistically significant: {}".format(model_comparison,p_value< 0.05))
          
      if np.mean(rv1) < np.mean(rv2):
          _,p_value = ttest_ind(rv1,rv2,equal_var=False,alternative='less')
          print("The GCN < {} difference is statistically significant: {}".format(model_comparison,p_value< 0.05))
    
    
if t_sne:
  device = torch.device('cuda')
  for corpus in ["mr", "ohsumed", "R8", "R52", "20ng"]:
      data = load_pytorch_geometric_data(corpus)
      data = data.to(device)
  
      best_model = torch.load("model_best_"+corpus+".pt").cuda()
      
      t_sne(best_model,data,corpus=corpus)
      
      torch.cuda.empty_cache()
   
   