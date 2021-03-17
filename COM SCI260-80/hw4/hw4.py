import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import time
import warnings
#warnings.filterwarnings('ignore')

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

class LSTM_Sentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(LSTM_Sentiment, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=False,bidirectional=True)

        # The linear layer that maps from hidden state space to sentiment space
        self.hidden2sentiment = nn.Linear(int(hidden_dim*2), num_classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        _, (last_hidden,_) = self.lstm(embeds)
        sentiment_scores = self.hidden2sentiment(last_hidden.permute(1,0,2).reshape(sentence.shape[1],-1).squeeze())
        
        return sentiment_scores
    
def train_step(model,data_iterator,loss_function,optimizer):
    running_loss = 0.0
    num_correct = 0
    num_incorrect = 0
    
    model.train()
    for text_object in tqdm(data_iterator):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = text_object.text[0]
        targets = text_object.label
        
        # Step 3. Run our forward pass.
        sentiment_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(sentiment_scores.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_correct = (sentiment_scores.argmax(1) == targets).type(torch.float).sum()
            num_correct += batch_correct
            num_incorrect += (len(targets)-batch_correct)
            
            running_loss += loss.detach().item()
            
    return (num_correct/(num_correct+num_incorrect)),running_loss/len(data_iterator)

def eval_step(model,data_iterator,loss_function):
    running_loss = 0.0
    num_correct = 0
    num_incorrect = 0
    
    model.eval()
    with torch.no_grad():
        for text_object in data_iterator:
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = text_object.text[0]
            targets = text_object.label

            # Step 3. Run our forward pass.
            sentiment_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(sentiment_scores.squeeze(), targets)

            batch_correct = (sentiment_scores.argmax(1) == targets).type(torch.float).sum()
            num_correct += batch_correct
            num_incorrect += (len(targets)-batch_correct)
            running_loss += loss.detach().item()
            
    return (num_correct/(num_correct+num_incorrect)),running_loss/len(data_iterator)


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=0.005)
parser.add_argument('--verbose',dest='verbose',help='True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False',action='store_true')
parser.add_argument('--save_fig',dest='save_fig',help='True to save the training curve figures (accuracy and loss curves), otherwise False',action='store_true')
parser.add_argument('--hidden',type=int,help="Number of hidden units for the LSTM",default=128)
args = parser.parse_args()

alpha = args.alpha
num_epochs = args.num_epochs
save_fig = args.save_fig
verbose = args.verbose
hidden = args.hidden

TEXT = data.Field(include_lengths=True)

# If you want to use English tokenizer from SpaCy, you need to install SpaCy and download its English model:
# pip install spacy
# python -m spacy download en_core_web_sm
# TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

LABEL = data.LabelField(dtype=torch.long)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

#TEXT.build_vocab(train_data)
# Here, you can also use some pre-trained embedding
TEXT.build_vocab(train_data,
                  vectors="glove.6B.100d",
                  unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

vocab_size = len(TEXT.vocab)
num_classes = len(LABEL.vocab)
HIDDEN_DIM = hidden
EMBEDDING_DIM = 100

model = LSTM_Sentiment(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, num_classes).to(device)
model.word_embeddings.load_state_dict({'weight':TEXT.vocab.vectors})

loss_function = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=alpha,weight_decay=0,momentum=0.9)

best_val_acc = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    train_acc,train_loss = train_step(model,train_iterator,loss_function,optimizer)
    val_acc,val_loss = eval_step(model,valid_iterator,loss_function)
    
    # save best model parameters (early stopping)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),"best_model.pt")
        
    # getting training and validatin history
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    if verbose:
        print("-"*30,epoch,"-"*30)
        print("Training Accuracy: {:.5f}".format(train_acc))
        print("Validation Accuracy: {:.5f}".format(val_acc))
        print("Training Error: {:.5f}".format(train_loss))
        print("Validation Error: {:.5f}".format(val_loss))
    
model.load_state_dict(torch.load("best_model.pt"))

val_acc,val_loss = eval_step(model,valid_iterator,loss_function)
test_acc,test_loss = eval_step(model,test_iterator,loss_function)

print("-"*30,epoch,"-"*30)
print("-"*30,epoch,"-"*30)
print("Validation Accuracy: {:.5f}".format(val_acc))
print("Validation Error: {:.5f}".format(val_loss))
print("Testing Accuracy: {:.5f}".format(test_acc))
print("Testing Error: {:.5f}".format(test_loss))

plt.figure(figsize=(10,10))
plt.plot(np.arange(num_epochs),train_acc_history,'r-',linewidth=5)
plt.plot(np.arange(num_epochs),val_acc_history,'b-',linewidth=5)
plt.plot([0,24],[0.80,0.80],'m--',linewidth=5)
plt.xlabel("Epoch",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.legend(["Training","Validation","80% Threshold"],fontsize=15)
if save_fig:
    plt.savefig("Accuracy_History"+".jpg")
    
plt.figure(figsize=(10,10))
plt.plot(np.arange(num_epochs),train_loss_history,'r-',linewidth=5)
plt.plot(np.arange(num_epochs),val_loss_history,'b-',linewidth=5)
plt.xlabel("Epoch",fontsize=15)
plt.ylabel("Cross Entropy Loss",fontsize=15)
plt.legend(["Training","Validation"],fontsize=15)
if save_fig:
    plt.savefig("Cross_Entropy_History"+".jpg")