# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from utils import *
from models import GCN
from metrics import *

import torch
import torch.nn.functional as F
import argparse 

parser = argparse.ArgumentParser()
#parser.add_argument('--model_choice', type=str, help='Model selection of FCNN (Fully Connected Neural Network) or CNN+FCNN (Convolutional Neural Network + CNN)', default='CNN+FCNN')
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=100)
parser.add_argument('--hidden', type=int, help='Neurons in hidden layer', default=25)

parser.add_argument('--print_every', type=int, help='Print metrics every "X" epochs', default=5)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=.1)
parser.add_argument('--verbose',dest='verbose',help='True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False',action='store_true')
parser.add_argument('--corpus', type=str, help='Corpus to train on (mr, ohsumed, R8, R52, 20ng)', default='mr')
#parser.add_argument('--save_fig',dest='save_fig',help='True to save the training curve figures (accuracy and loss curves), otherwise False',action='store_true')
args = parser.parse_args()

 

alpha = args.alpha
#model_choice = args.model_choice
num_epochs = args.num_epochs
hidden = args.hidden
#save_fig = args.save_fig
verbose = args.verbose
print_every = args.print_every
corpus = args.corpus


best_val_acc = 0


save_path = "model_best_"+corpus+".pt"

device = torch.device('cuda')
data = load_pytorch_geometric_data(corpus)
data = data.to(device)

model = GCN(data,hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=alpha)

best_val_acc = 0
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        
        _,pred = model(data).max(dim=1)

        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_acc = correct/int(data.val_mask.sum())

        correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        train_acc = correct/int(data.train_mask.sum())
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model,save_path)

    if epoch % print_every == 0 and verbose==True:

        print("Epoch {}".format(epoch))
        print("Loss {:.4f}".format(loss))
        print('Validation Accuracy: {:.6f}'.format(val_acc))
        print("Training Accuracy: {:.6f}".format(train_acc))
        print("-"*50)
        

del model
torch.cuda.empty_cache()

best_model = torch.load(save_path)
best_model.eval()
    
test_idx = data.test_mask
_,y_pred = best_model(data).max(dim=1)
y_pred = y_pred[test_idx].cpu()


y_true = data.y[test_idx].cpu().numpy()
labels = np.unique(y_true)

evaluation_report(y_true,y_pred,labels)
