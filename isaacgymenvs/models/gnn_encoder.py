import os.path

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from model import *
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './checkpoint'
if_model = True
save_dir = '../runs_tac/'

task_name = 'ball_gnn_binary_64channels'
object_name = 'dataset'

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 30

tac_list = []
tactile_pos_list = []
object_pos_list = []

for i in range (61,67):
    index = i *100
    with open(save_dir+object_name+'_%d.pkl'%index,'rb') as f:
        data = pickle.load(f)
    tactile_log = np.array(data['tactile']).reshape(-1,653)
    tactile_pos_log = np.array(data['tac_pose']).reshape(-1,653,3)
    object_pos_log = np.array(data['object_pos']).reshape(-1,6)
    for j in range(tactile_log.shape[0]):
        tactile = tactile_log[j]
        object_pos = object_pos_log[j]
        tactile_pos = tactile_pos_log[j]
        if tactile[tactile>0].shape[0] >5:
            tac_list.append(tactile)
            object_pos_list.append(object_pos)
            tactile_pos_list.append(tactile_pos)

tac = torch.tensor(np.array(tac_list), device=device, dtype=torch.float32)
pos = torch.tensor(np.array(tactile_pos_list), device=device, dtype=torch.float32) *100
y = torch.tensor(np.array(object_pos_list), device=device, dtype=torch.float32) *100

# tac /= tac.max(1,keepdim=True)[0]
tac = tac.view(-1,653,1)

tactile_dataset = []
for i in range(tac.shape[0]):

    data = Data(x=tac[i,tac[i,:,0]!=0,:],pos=pos[i,tac[i,:,0]!=0,:],y=y[i].view(1,-1))
    tactile_dataset.append(data)

m=len(tactile_dataset)

train_data, val_data = random_split(tactile_dataset, [int(m*0.8), m-int(m*0.8)])

train_loader = DataLoader(train_data, batch_size=32)
valid_loader = DataLoader(val_data, batch_size=32)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the network

if if_model:
    classifier = torch.load(os.path.join(model_dir,task_name+'_encoder.pt'))
else:
    classifier = GNNEncoder(device=device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-05)

# Move both the encoder and the decoder to the selected device
classifier.to(device)

### Training function
def train_epoch(classifier, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    classifier.train()

    total_loss = 0
    train_loss = []
    for data in dataloader:
        optimizer.zero_grad()  # Clear gradients.
        logits = classifier(data.x, data.pos, data.batch)  # Forward pass.
        loss = loss_fn(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

@torch.no_grad()
def test_epoch(classifier, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    total_loss = 0
    test_loss= []
    for data in dataloader:
        logits = classifier(data.x, data.pos, data.batch)  # Forward pass.
        loss = loss_fn(logits, data.y)  # Loss computation.
        total_loss += loss.item() * data.num_graphs
        test_loss.append(loss.detach().cpu().numpy())
    return np.mean(test_loss)

if not if_model:

    num_epochs = 100
    diz_loss = {'train_loss': [], 'val_loss': []}
    Accuracy_list = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(classifier, train_loader, loss_fn, optimizer)
        val_loss = test_epoch(classifier, valid_loader, loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)

    torch.save(classifier, os.path.join(model_dir, task_name + '_encoder.pt'))

    np.save(os.path.join(model_dir, task_name + '_train_loss'),np.array(diz_loss['train_loss']))
    np.save(os.path.join(model_dir, task_name + '_val_loss'),np.array(diz_loss['val_loss']))

    # # Plot losses
    # plt.figure(figsize=(10,8))
    # plt.semilogy(diz_loss['train_loss'], label='Train')
    # plt.semilogy(diz_loss['val_loss'], label='Valid')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')
    # #plt.grid()
    # plt.legend()
    # #plt.title('loss')
    # plt.show()
    #
    # # Plot Accuracy
    # plt.figure(figsize=(10,8))
    # plt.plot(Accuracy_list, label='Test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Accuracy')
    # #plt.grid()
    # plt.legend()
    # #plt.title('loss')
    # plt.show()
if if_model:
    num_epochs = 1
    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        val_loss = test_epoch(classifier, valid_loader, loss_fn)
        print('\n EPOCH {}/{} \t val loss {}'.format(epoch + 1, num_epochs, val_loss))
        diz_loss['val_loss'].append(val_loss)
