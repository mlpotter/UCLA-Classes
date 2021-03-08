import os
import random
import numpy as np

import torch
import pandas as pd
import argparse

from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertConfig

from tqdm import tqdm

from torch.optim import Adam

def corpus_loader(dataset='20ng'):

    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('/mnt/WDMyBook/ilkay/data/' + dataset + '.txt', 'r')
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    doc_content_list = []
    f = open('/mnt/WDMyBook/ilkay/data/corpus/' + dataset + '.clean.txt', 'r')
    for line in f.readlines():
        doc_content_list.append(line.strip())
    f.close()

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)

    random.shuffle(train_ids)

    #train_ids = train_ids[:int(0.2 * len(train_ids))]

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)

    random.shuffle(test_ids)

    ids = train_ids + test_ids


    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size


    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
        
    train_labels = []

    test_labels = []

    val_labels = []

    for i in range(len(shuffle_doc_words_list)):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]

        if i < train_size:
            train_labels.append(label)

        elif i >= train_size and i<(train_size+val_size):
            val_labels.append(label)

        else:
            test_labels.append(label)
            
    factorized_labels = list(pd.factorize(train_labels + val_labels + test_labels)[0])
    
    train_texts,train_labels = shuffle_doc_words_list[:train_size], factorized_labels[:train_size]
    val_texts,val_labels = shuffle_doc_words_list[train_size:(train_size+val_size)], factorized_labels[train_size:(train_size+val_size)]
    test_texts,test_labels = shuffle_doc_words_list[(train_size+val_size):],factorized_labels[(train_size+val_size):]
    
    return train_texts,val_texts,test_texts,train_labels,val_labels,test_labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def train_step(model,train_dataloader,optimizer):
    model.train()
    num_correct = 0
    num_incorrect = 0
    running_loss = 0
    
    for batch in tqdm(train_dataloader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            y_pred = outputs[1].argmax(1)
            batch_correct = (y_pred==labels).type(torch.float).sum().item()
            num_correct += batch_correct
            num_incorrect += (labels.shape[0]-batch_correct)
            running_loss += loss.detach().item()
    
    return num_correct/(num_correct+num_incorrect),running_loss/len(train_dataloader)

def validation_step(model,val_dataloader):
    model.eval()
    num_correct = 0
    num_incorrect = 0
    running_loss = 0
 
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            y_pred = outputs[1].argmax(1)
            batch_correct = (y_pred==labels).type(torch.float).sum().item()
            num_correct += batch_correct
            num_incorrect += (labels.shape[0]-batch_correct)
            running_loss += loss.detach().item()

    return num_correct/(num_correct+num_incorrect),running_loss/len(val_dataloader)

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--print_every', type=int, help='Print metrics every "X" echs', default=5)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=1e-5)
parser.add_argument('--verbose',dest='verbose',help='True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False',action='store_true')
parser.add_argument('--corpus', type=str, help='Corpus to train on (mr, ohsumed, R8, R52, 20ng)', default='mr')
parser.add_argument('--batch_size', type=int, help='Batch Size to use for dataloaders', default=8)
parser.add_argument('--pretrained',dest='pretrained',help='True to use pretrained DistilBert, otherwise False',action='store_true')

args = parser.parse_args()

 

 
alpha = args.alpha
num_epochs = args.num_epochs
verbose = args.verbose
print_every = args.print_every
corpus = args.corpus
batch_size = args.batch_size
pretrained = args.pretrained


train_texts,val_texts,test_texts,train_labels,val_labels,test_labels = corpus_loader(corpus)
num_labels = len(np.unique(train_labels+val_labels+test_labels))

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',max_position_embedding=2048)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

configuration = DistilBertConfig(num_labels=num_labels,max_position_embeddings=2048)#n_layers=3,sinusoidal_pos_embds=True,hidden_dim=1536,dim=420)
if pretrained:
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=num_labels)
else:
  model = DistilBertForSequenceClassification(configuration)#.from_pretrained('distilbert-base-uncased',num_labels=num_labels)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size)
test_loader = DataLoader(test_dataset,batch_size=batch_size)

optim = Adam(model.parameters(), lr=alpha)

best_val_acc = 0
for epoch in range(num_epochs):
    train_acc,train_loss = train_step(model,train_loader,optim)
    val_acc,val_loss = validation_step(model,val_loader)
    
    if best_val_acc < val_acc:
        model.save_pretrained(r"/home/ilkay/Michael_Potter_Final_Project/BERT_best_"+corpus)
        best_val_acc = val_acc
    
    if verbose:
        if ((epoch+1) % print_every) == 0:
            print("-"*25,"epoch {}".format(epoch),"-"*25)
            print("Train Accuracy: {:.7f}".format(train_acc))
            print("Validation Accuracy: {:.7f}".format(val_acc))

model = model.from_pretrained(r"/home/ilkay/Michael_Potter_Final_Project/BERT_best_"+corpus)
model.to(device)
model.eval()
test_acc,test_loss = validation_step(model,test_loader)
print("Testing Accuracy {:.7f}".format(test_acc))

torch.cuda.empty_cache()