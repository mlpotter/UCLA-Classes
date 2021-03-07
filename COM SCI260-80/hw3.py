# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 06:48:52 2021

@author: lpott
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
import argparse
import matplotlib.pyplot as plt

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

class fc_net(nn.Module):
    def __init__(self,activation=True):
        super(fc_net, self).__init__()
        
        if activation==True:
            self.fc_layers = nn.Sequential(
                nn.Linear(32*32*3,1536), #layer 1
                nn.ReLU(inplace=True),
                nn.Linear(1536,1536), #layer 2
                nn.ReLU(inplace=True),
                nn.Linear(1536,1536), #layer 3
                nn.ReLU(inplace=True),
                nn.Linear(1536,1536), #layer 4
                nn.ReLU(inplace=True),
                nn.Linear(1536,768), #layer 5
                nn.ReLU(inplace=True),
                nn.Linear(768,384), # layer 6
                nn.ReLU(inplace=True),
                nn.Linear(384,10) # layer 7
            )

        if activation==False:
            self.fc_layers = nn.Sequential(
                nn.Linear(32*32*3,1536), #layer 1
                nn.Linear(1536,1536), #layer 2
                nn.Linear(1536,1536), #layer 3
                nn.Linear(1536,1536), #layer 4 
                nn.Linear(1536,768), #layer 5
                nn.Linear(768,384), #layer 6
                nn.Linear(384,10) #layer 7
                ) 
            
        print("Model Constructed")
        print("Activations are {}".format(["Off","On"][int(activation)]))

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,-1)
        x = self.fc_layers(x)
        
        return x
    
class cnn_net(nn.Module):
    def __init__(self,activation=True):
        super(cnn_net, self).__init__()
        if activation==True:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=2), #layer 1
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 124, kernel_size=3, padding=1,stride=1), #layer 2
                nn.ReLU(inplace=True),
                nn.Conv2d(124, 256, kernel_size=3, padding=1,stride=1), #layer 3
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, kernel_size=3, padding=1,stride=2), #layer 4
                nn.ReLU(inplace=True),
            )
            
            self.fc_layers = nn.Sequential(
                nn.Linear(4096,4096), #layer 1
                nn.ReLU(inplace=True),
                nn.Linear(4096,2048), #layer 2
                nn.ReLU(inplace=True),
                nn.Linear(2048,10), #layer 3
            )

        if activation==False:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=2), #layer 1
                nn.Conv2d(64, 124, kernel_size=3, padding=1,stride=1), #layer 2
                nn.Conv2d(124, 256, kernel_size=3, padding=1,stride=2), #layer 3
                nn.Conv2d(256, 64, kernel_size=3, padding=1,stride=1), #layer 4
            )
            
            self.fc_layers = nn.Sequential(
                nn.Linear(4096,4096), #layer 1
                nn.Linear(4096,2048), #layer 2
                nn.Linear(2048,10), #layer 3
            )
            
        print("Model Constructed")
        print("Activations are {}".format(["Off","On"][int(activation)]))

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.cnn_layers(x)
        x = x.view(B,-1)
        x = self.fc_layers(x)
        
        return x
    
def train_step(model,train_loader,criterion,optimizer):
    running_loss = 0.0
    running_correct = 0
    running_incorrect = 0
    
    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
            correct_predictions = (outputs.argmax(1) == labels.cuda()).type(torch.float)
            running_loss += loss.item()
            running_correct += correct_predictions.sum().item()
            running_incorrect += (1-correct_predictions).sum().item()
        
    return (running_loss/len(train_loader),running_correct/(running_correct+running_incorrect))

def validation_step(model,test_loader,criterion):
    running_correct = 0
    running_incorrect = 0
    running_loss = 0.0

    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())

            
            correct_predictions = (outputs.argmax(1) == labels.cuda()).type(torch.float)

            # print statistics
            running_correct += correct_predictions.sum().item()
            running_incorrect += (1-correct_predictions).sum().item()
            running_loss += loss.item()  
            
    return (running_loss/len(test_loader),running_correct/(running_correct+running_incorrect))

def fit_model(model,criterion,optimizer,num_epochs,train_loader,test_loader,filename="model",verbose=False):
    num_epochs = num_epochs
    best_eval_accuracy = 0
    training_accuracy = []
    training_error = []
    validation_accuracy = []
    validation_error = []
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        (epoch_train_loss,epoch_train_accuracy) = train_step(model,train_loader,criterion,optimizer)
        (epoch_val_loss,epoch_val_accuracy) = validation_step(model,test_loader,criterion)

        if best_eval_accuracy < epoch_val_accuracy:
            best_eval_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(),filename+".pt")
            
        if verbose==True:
            print("-"*25 + "Epoch {}".format(epoch) + "-"*25)
            print("Train Cross Entropy Loss {:.4f}".format(epoch_train_loss))
            print("Validation Cross Entropy Loss {:.4f}".format(epoch_val_loss))
    
            print("Training Accuracy {:.3f}".format(epoch_train_accuracy))
            print("Validation Accuracy {:.3f}".format(epoch_val_accuracy))
        
        training_accuracy.append(epoch_train_accuracy)
        training_error.append(epoch_train_loss)
        validation_accuracy.append(epoch_val_accuracy)
        validation_error.append(epoch_val_loss)
        
    print("-"*50 + "\n")
    print('Finished Training')
    solver = {"best_model":model.load_state_dict(torch.load(filename+".pt")),
             "train_acc_history":training_accuracy,
             "train_error_history":training_error,
             "val_acc_history":validation_accuracy,
             "val_error_history":validation_error,
             "num_epochs":num_epochs,
             "model_name":filename}
    print("Best Accuracy: {:.4f}".format(best_eval_accuracy))
    return solver

def visualize(solver,threshold=0.85,savefig=False):
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(solver['num_epochs']),solver['train_acc_history'],'b-')
    plt.plot(np.arange(solver['num_epochs']),solver['val_acc_history'],'r-')
    plt.plot([0,solver['num_epochs']],[threshold,threshold],'m--',linewidth=4)
    plt.legend(['training accuracy','validation accuracy','85% Mark'],fontsize=10)
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    if savefig == True:
        plt.savefig(solver['model_name']+"_acc_history"+".jpg")
    
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(solver['num_epochs']),solver['train_error_history'],'b-')
    plt.plot(np.arange(solver['num_epochs']),solver['val_error_history'],'r-')
    plt.legend(['training loss','validation loss'],fontsize=10)
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Cross Entropy Loss',fontsize=15)
    if savefig==True:
        plt.savefig(solver['model_name']+"_error_history"+".jpg")
    
batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)



parser = argparse.ArgumentParser()
parser.add_argument('--model_choice', type=str, help='Model selection of FCNN (Fully Connected Neural Network) or CNN+FCNN (Convolutional Neural Network + CNN)', default='CNN+FCNN')
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=100)
parser.add_argument('--activation',dest='activation',help='True to use ReLU activations after each hidden layer, otherwise False',action='store_true')
parser.add_argument('--alpha', type=float, help='Learning Rate', default=.05)
parser.add_argument('--verbose',dest='verbose',help='True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False',action='store_true')
parser.add_argument('--save_fig',dest='save_fig',help='True to save the training curve figures (accuracy and loss curves), otherwise False',action='store_true')
parser.add_argument('--threshold', type=float, help='Accuracy threshold for model to beat (will be plotted as line on accuracy learning curves)', default=.85)
args = parser.parse_args()

alpha = args.alpha
model_choice = args.model_choice
num_epochs = args.num_epochs
save_fig = args.save_fig
verbose = args.verbose
activations = args.activation
threshold = args.threshold

print("Activations: ",activations)
print("Number of Epochs: ",num_epochs)
print("Save Figures: ",save_fig)
print("Learning Rate: ",alpha)
print("Model Choice: ",model_choice)
print("Verbose: ",verbose)

if model_choice == "CNN+FCNN":
    model = cnn_net(activation=activations).cuda()
if model_choice == "FCNN":
    model = fc_net(activation=activations).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=alpha)

solver = fit_model(model,criterion,optimizer,num_epochs,train_loader,test_loader,model_choice+"_"+str(activations),verbose=verbose)

visualize(solver,threshold=threshold,savefig=save_fig)