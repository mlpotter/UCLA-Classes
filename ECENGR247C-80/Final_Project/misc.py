# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:26:11 2021

@author: Michael Potter
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def t_sne(model,data,corpus=""):
    model.eval()
    
    test_idx = data.test_mask
    
    y_true = data.y[test_idx].cpu().detach().numpy()
    
    x,edge_index,edge_weight = data.x,data.edge_index,data.edge_weight
    l1 = model.conv1(x,edge_index,edge_weight)
    l2 = model.conv2(F.relu(l1),edge_index,edge_weight)[test_idx].cpu().detach().numpy()
    l1 = l1[test_idx].cpu().detach().numpy()
    
    l1_embedded = TSNE(n_components=2).fit_transform(l1)
    l2_embedded = TSNE(n_components=2).fit_transform(l2)
    
    plt.figure()
    plt.scatter(l1_embedded[:,0],l1_embedded[:,1],c=y_true,cmap='jet')
    plt.title("corpus+Layer 1 Embeddings")
    plt.show()
    
    plt.figure()
    plt.scatter(l2_embedded[:,0],l2_embedded[:,1],c=y_true,cmap='jet',alpha=0.5)
    plt.title(corpus+"Layer 2 Embeddings")
    plt.show()
    
    torch.cuda.empty_cache()