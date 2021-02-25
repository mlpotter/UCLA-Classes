# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:12:33 2021

@author: Michael Potter
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from utils import *

from metrics import *
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--verbose',dest='verbose',help='True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False',action='store_true')
parser.add_argument('--corpus', type=str, help='Corpus to train on (mr, ohsumed, R8, R52, 20ng)', default='mr')
args = parser.parse_args()

verbose = args.verbose
corpus = args.corpus


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(corpus)

adj = adj + sp.eye(adj.shape[0])

adj = adj[:,(train_size):(train_size+(adj.shape[0]-test_size))]

y_train = y_train.argmax(1)
y_test = y_test.argmax(1)
y_val = y_val.argmax(1)

best_val_acc = 0

penalties = ['l1','l2']
C = np.linspace(0.001,1,25)
iterations = [100,1000]#,10000]

for iteration in iterations:
    for penalty in penalties:
        for c in C:

            lr = LogisticRegression(penalty=penalty,C=c,max_iter=iteration,solver='liblinear')
            lr.fit(adj[train_mask],y_train[train_mask])
            
            y_pred = lr.predict(adj[val_mask])
            val_acc = np.mean(y_pred==y_val[val_mask])
            
            if verbose:
                print("-"*50)
                print("C={} \t Penalty={} \t Iteration={}".format(c,penalty,iteration))
                print("Accuracy: {}".format(val_acc))

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                dump(lr,'logisticmodel'+corpus+'.joblib')
 
best_model = load('logisticmodel'+corpus+'.joblib') 

y_pred = best_model.predict(adj[test_mask])

evaluation_report(y_pred,y_test[test_mask],np.unique(y_test[test_mask]))