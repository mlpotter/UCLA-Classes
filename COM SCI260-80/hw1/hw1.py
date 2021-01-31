#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse


## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA
torch.manual_seed(123)
np.random.seed(123)

train_data = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))

class Logistic_Regression(nn.Module):
    def __init__(self):
        super(Logistic_Regression,self).__init__()
        self.linear_layer = nn.Linear(28*28,1)
        self.sigmoid_layer = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid_layer(self.linear_layer(x))

class Support_Vector_Machine(nn.Module):
    def __init__(self):
        super(Support_Vector_Machine,self).__init__()
        self.linear_layer = nn.Linear(28*28,1)

    def forward(self,x):
        return self.linear_layer(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_choice', type=str, help='Model selection of LR (Logistic Regression) or SVM (Support Vector Machine)', default='LR')
    parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=10)
    parser.add_argument('--momentum', type=float, help='Momentum for SGD [0,1)', default=0.0)
    parser.add_argument('--alpha', type=float, help='Learning Rate', default=.5)
    parser.add_argument('--save_fig',type=bool,help='True to save the training curve figure, otherwise False',default=False)
    args = parser.parse_args()
    
    alpha = args.alpha
    beta = args.momentum
    model_choice = args.model_choice
    num_epochs = args.num_epochs
    save_fig = args.save_fig
    
    models = {"LR":Logistic_Regression,"SVM":Support_Vector_Machine}
    loss_fns = {"LR":nn.BCELoss(),"SVM":lambda y_hat,y: torch.mean(torch.clamp(1-y_hat*y,min=0))}
    
    model = models[model_choice]()
    model_loss = loss_fns[model_choice]
    model_optimizer = torch.optim.SGD(model.parameters(),lr=alpha,momentum=beta)
    
    # The number of epochs is at least 10, you can increase it to achieve better performance
    num_epochs = 10

    # Training the Model
    epoch_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            #Convert labels from 0,1 to -1,1
            labels = 2*(labels.float()-0.5)

            ## TODO 
            # zero the parameter gradients
            model_optimizer.zero_grad()

            # change label accordingly...
            labels = (labels/2 + .5) if model_choice=="LR" else labels

            # forward + backward + optimize LR
            outputs = model(images)
            loss = model_loss(outputs.squeeze(),labels)
            loss.backward()
            model_optimizer.step()

            total_loss = total_loss + loss.detach().item()

        ## Print your results every epoch
        epoch_loss.append(total_loss/(len(train_loader)))
        print("Epoch {}: Loss={:.5f}".format(epoch+1,total_loss/len(train_loader)))
        
    if save_fig:
        plt.figure(figsize=(15,15))
        filename = '{} + SGD-{:.3f}.png'.format(model_choice,beta)
        plt.plot(np.arange(len(epoch_loss)),epoch_loss,linewidth=10)
        plt.xlabel("Epoch Number",fontsize=50)
        plt.ylabel("Epoch Loss",fontsize=50)
        plt.title(filename[:(len(filename)-4)],fontsize=50)
        plt.xticks(ticks=np.arange(len(epoch_loss)),labels=np.arange(len(epoch_loss))+1,fontsize=50); plt.yticks(fontsize=50)
        plt.tight_layout()
        plt.savefig(filename)
        
    # Test the Model
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.view(-1, 28*28)

        ## Put your prediction code here, currently use a random prediction
        with torch.no_grad():
            prediction = (model(images)>=.5).float() if model_choice == "LR" else (model(images)>=0).float()

        correct += (prediction.view(-1).long() == labels).sum()
        total += images.shape[0]
    print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
        
if __name__ == '__main__':
    main()