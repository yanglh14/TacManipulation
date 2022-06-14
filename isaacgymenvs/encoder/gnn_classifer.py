import os.path

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from gnn import *
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './models'
if_model = True
save_dir = '../tac_data/'

task_name = 'final_multi_2'
object_name = 'tac_data_multi'


with open(save_dir+object_name+'.pkl','rb') as f:
    d = pickle.load(f)
tac = abs(torch.tensor(d['tactile'],dtype=torch.float32, device=device)[:,:,2])
pos = torch.tensor(d['tac_pose'],dtype=torch.float32, device=device)
y = torch.tensor(d['class'],dtype=torch.long, device=device)

tac /= tac.max(1,keepdim=True)[0]
tac = tac.view(10000,-1,1)

tactile_dataset = []
for i in range(tac.shape[0]):

    data = Data(x=tac[i,tac[i,:,0]!=0,:],pos=pos[i,tac[i,:,0]!=0,:],y=y[i])
    tactile_dataset.append(data)

m=len(tactile_dataset)

train_data, val_data = random_split(tactile_dataset, [int(m-m*0.2), int(m*0.2)])

train_loader = DataLoader(train_data, batch_size=32)
valid_loader = DataLoader(val_data, batch_size=32)

### Define the loss function
loss_fn = nn.CrossEntropyLoss()

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the network

if if_model:
    classifier = torch.load(os.path.join(model_dir,task_name+'_classifier.pt'))
else:
    classifier = PointNet(device=device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-05)

# Move both the encoder and the decoder to the selected device
classifier.to(device)

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.x, data.pos, data.batch)  # Forward pass.
        loss = loss_fn(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.x, data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)

### Training function
def train_epoch(classifier, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    classifier.train()

    total_loss = 0
    for data in dataloader:
        optimizer.zero_grad()  # Clear gradients.
        logits = classifier(data.x, data.pos, data.batch)  # Forward pass.
        loss = loss_fn(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def test_epoch(classifier, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    total_loss = 0
    for data in dataloader:
        logits = classifier(data.x, data.pos, data.batch)  # Forward pass.
        loss = loss_fn(logits, data.y)  # Loss computation.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)

def accuracy_epoch(classifier, device, dataloader,one_class=None):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        correct = 0
        total = 0

        for data in dataloader:

            # Encode data
            if one_class != None:
                class_ = torch.zeros_like(data.batch)
                for i in range(data.batch.shape[0]):
                    class_[i] = ((data.y==one_class)[data.batch[i]] == True)

                data.x = data.x[class_==True]
                data.batch = data.batch[class_==True]
                data.pos = data.pos[class_==True]
                data.y = data.y[data.y==one_class]

                uni = torch.unique(data.batch)
                j = 0
                for i in uni:
                    data.batch[data.batch==i] = j
                    j += 1
            if data.y.size(0)>0:
                encoded_data = classifier(data.x, data.pos, data.batch)
                pred = encoded_data.argmax(dim=-1)

                correct += int((pred == data.y).sum())
                total += data.y.size(0)

    Accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {Accuracy} %')

    return Accuracy

def probablity_dis(classifier, device, dataloader,one_class=None):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad(): # No need to track the gradients
        First = True
        for data in dataloader:
            # Encode data
            if one_class != None:
                class_ = torch.zeros_like(data.batch)
                for i in range(data.batch.shape[0]):
                    class_[i] = ((data.y==one_class)[data.batch[i]] == True)

                data.x = data.x[class_==True]
                data.batch = data.batch[class_==True]
                data.pos = data.pos[class_==True]
                data.y = data.y[data.y==one_class]

                uni = torch.unique(data.batch)
                j = 0
                for i in uni:
                    data.batch[data.batch==i] = j
                    j += 1

            if data.y.size(0)>0:
                encoded_data = classifier(data.x, data.pos, data.batch)

                if First:
                    predicted_list = encoded_data.cpu().numpy()
                    First = False
                else:
                    predicted_list = np.append(predicted_list,encoded_data.cpu().numpy(),axis=0)
        probablity = np.average(predicted_list,axis=0)

        return probablity

if not if_model:

    num_epochs = 50
    diz_loss = {'train_loss': [], 'val_loss': []}
    Accuracy_list = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(classifier, train_loader, loss_fn, optimizer)
        val_loss = test_epoch(classifier, valid_loader, loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        Accuracy = accuracy_epoch(classifier, device, valid_loader)
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        Accuracy_list.append(Accuracy)

    torch.save(classifier, os.path.join(model_dir, task_name + '_classifier.pt'))

    np.save(os.path.join('./data', task_name + '_classifier_train_loss'),np.array(diz_loss['train_loss']))
    np.save(os.path.join('./data', task_name + '_classifier_val_loss'),np.array(diz_loss['val_loss']))
    np.save(os.path.join('./data', task_name + '_classifier_accuracy'),np.array(Accuracy_list))

    # Plot losses
    plt.figure(figsize=(10,8))
    plt.semilogy(diz_loss['train_loss'], label='Train')
    plt.semilogy(diz_loss['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10,8))
    plt.plot(Accuracy_list, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.show()

if if_model:
    accuracy_list = []
    for i in range(10):
        accuracy = accuracy_epoch(classifier, device, valid_loader,one_class=i)
        accuracy_list.append(accuracy)

    fig, ax = plt.subplots()
    labels = list(range(10))
    ax.bar(labels, accuracy_list, 0.35, label='Test')

    ax.set_xlabel('Object Class')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    plt.show()

    pro_list=[]
    for i in range(10):
        probablity = probablity_dis(classifier, device, valid_loader, one_class=i)
        pro_list.append(probablity)
        fig, ax = plt.subplots()
        labels = list(range(10))
        ax.bar(labels, probablity, 0.35, label='Test')
        ax.set_xlabel('Object Class')
        ax.set_ylabel('Average Probability')
        ax.set_title(f'Object{i} Prediction Probability')
        ax.legend()
        plt.show()
    np.save(os.path.join('./data', 'pro_'+task_name),np.array(pro_list))
