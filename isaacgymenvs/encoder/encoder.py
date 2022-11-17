import os.path

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from isaacgymenvs.encoder.model import *
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

class encoder():

    def __init__(self):
        # Check if the GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')
        self.dataset_num = 50
        self.model_dir = './checkpoint_new'
        self.if_model = False
        self.save_dir = '../runs_tac2/'
        self.model_type = 'cnn'
        self.channels = 32
        self.task_name = 'ball_%s_%d_2'%(self.model_type,self.channels)
        self.object_name = 'dataset'
        ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()
        print(self.model_type)
        ### Set the random seed for reproducible results
        torch.manual_seed(44)
        np.random.seed(44)
        ### Initialize the network

        if self.if_model:
            self.model = torch.load(os.path.join(self.model_dir, self.task_name + '_encoder.pt'))

        else:
            if self.model_type == 'gnn':
                self.model = GNNEncoderB(device=self.device, pos_pre_bool= False, channels = self.channels)
            elif self.model_type == 'gnn_pre':
                self.model = GNNEncoderB(device=self.device, pos_pre_bool= True, channels = self.channels)
            elif self.model_type == 'cnn':
                self.model = CNNEncoder(encoded_space_dim=6)
            elif self.model_type == 'mlp':
                self.model = MLPEncoder(encoded_space_dim=6)
            elif self.model_type == 'gcn':
                self.model = GCNEncoder(device=self.device, channels = self.channels)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-05)

        # Move both the encoder and the decoder to the selected device
        self.model.to(self.device)

    def run(self):
        if self.if_model:
            self.test()
        else:
            self.train()

    def prepare_dataset(self,index_1,index_2):
        tac_list = []
        tactile_pos_list = []
        object_pos_list = []
        object_pos_pre_list = []

        for i in range(index_1+1,index_2+1):
            index = i *100
            with open(self.save_dir+self.object_name+'_%d.pkl'%index,'rb') as f:
                data = pickle.load(f)
            tactile_log = np.array(data['tactile']).reshape(-1,653)
            tactile_pos_log = np.array(data['tac_pose']).reshape(-1,653,3)
            object_pos_log = np.array(data['object_pos']).reshape(-1,6)
            object_pos_pre_log = np.array(data['object_pos']).reshape(-1,6)

            for j in range(tactile_log.shape[0]):
                tactile = tactile_log[j]
                object_pos = object_pos_log[j]
                tactile_pos = tactile_pos_log[j]
                object_pos_pre = object_pos_pre_log[j]

                tac_list.append(tactile)
                object_pos_list.append(object_pos)
                tactile_pos_list.append(tactile_pos)
                object_pos_pre_list.append(object_pos_pre)

        if self.model_type == 'cnn':
            tac_list = self.tactile_process(tac_list)

        tac = torch.tensor(np.array(tac_list), device=self.device, dtype=torch.float32)
        pos = torch.tensor(np.array(tactile_pos_list), device=self.device, dtype=torch.float32) *100
        y = torch.tensor(np.array(object_pos_list), device=self.device, dtype=torch.float32) *100
        object_pos_pre = torch.tensor(np.array(object_pos_pre_list), device=self.device, dtype=torch.float32) *100

        # tac /= tac.max(1,keepdim=True)[0]
        batch_size = 32
        if self.model_type == 'gnn' or self.model_type == 'gnn_pre' or self.model_type == 'gcn':
            tac = tac.view(-1, 653, 1)

            tactile_dataset = []
            for i in range(tac.shape[0]):
                if tac[i,tac[i,:,0]!=0,:].shape[0] > 5:
                    data = Data(x=tac[i,tac[i,:,0]!=0,:],pos=pos[i,tac[i,:,0]!=0,:],y=y[i].view(1,-1), pos_pre = object_pos_pre[i,:].view(1,-1))

                    tactile_dataset.append(data)

            m=len(tactile_dataset)

            train_data, val_data = random_split(tactile_dataset, [int(m*0.8), m-int(m*0.8)])

            self.train_loader = DataLoader(train_data, batch_size=batch_size)
            self.valid_loader = DataLoader(val_data, batch_size=batch_size)

        elif self.model_type == 'cnn':
            x = tac.reshape(-1, 1, 36, 36)
            tactile_dataset = TensorDataset(x, y)

            m = len(tactile_dataset)

            train_data, val_data = random_split(tactile_dataset, [int(m * 0.8), m - int(m * 0.8)])

            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
            self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

        elif self.model_type == 'mlp':
            x = tac.reshape(-1, 653)
            tactile_dataset = TensorDataset(x, y)

            m = len(tactile_dataset)

            train_data, val_data = random_split(tactile_dataset, [int(m * 0.8), m - int(m * 0.8)])

            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
            self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    ### Training function
    def train_epoch(self):
        # Set train mode for both the encoder and the decoder
        self.model.train()


        total_loss =0
        total_num = 0
        if self.model_type == 'gnn' or self.model_type == 'gnn_pre' or self.model_type == 'gcn':
            for i in range(0, self.dataset_num):
                self.prepare_dataset(i, (i + 1))

                for data in self.train_loader:
                    self.optimizer.zero_grad()  # Clear gradients.
                    logits = self.model(data.x, data.pos, data.batch, data.pos_pre)  # Forward pass.
                    loss = self.loss_fn(logits, data.y)  # Loss computation.
                    loss.backward()  # Backward pass.
                    self.optimizer.step()  # Update model parameters.
                    total_loss += loss.item() * data.num_graphs
                total_num += self.train_loader.dataset.__len__()

        elif self.model_type == 'cnn' or self.model_type == 'mlp':
            for i in range(0, self.dataset_num):
                self.prepare_dataset(i, (i + 1))

                for features, labels in self.train_loader:
                    self.optimizer.zero_grad()  # Clear gradients.
                    logits = self.model(features)  # Forward pass.
                    loss = self.loss_fn(logits, labels)  # Loss computation.
                    loss.backward()  # Backward pass.
                    self.optimizer.step()  # Update model parameters.
                    total_loss += loss.item() * labels.shape[0]
                total_num += self.train_loader.dataset.__len__()

        return total_loss/ total_num

    @torch.no_grad()
    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.model.eval()
        total_loss =0
        total_num = 0

        if self.model_type == 'gnn' or self.model_type == 'gnn_pre' or self.model_type == 'gcn':
            for i in range(0, self.dataset_num):
                self.prepare_dataset(i, (i + 1))

                for data in self.valid_loader:
                    logits = self.model(data.x, data.pos, data.batch, data.pos_pre)  # Forward pass.
                    loss = self.loss_fn(logits, data.y)  # Loss computation.
                    total_loss += loss.item() * data.num_graphs
                total_num += self.valid_loader.dataset.__len__()

        elif self.model_type == 'cnn' or self.model_type == 'mlp':
            for i in range(0, self.dataset_num):
                self.prepare_dataset(i, (i + 1))

                for features, labels in self.valid_loader:
                    logits = self.model(features)  # Forward pass.
                    loss = self.loss_fn(logits, labels)  # Loss computation.
                    total_loss += loss.item() * labels.shape[0]
                total_num += self.valid_loader.dataset.__len__()

        return total_loss/ total_num

    def train(self):

        num_epochs = 30
        diz_loss = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            # train_loss_list = []
            # test_loss_list = []

            train_loss = self.train_epoch()
            val_loss = self.test_epoch()
            # train_loss_list.append(train_loss)
            # test_loss_list.append(val_loss)
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))

            # train_loss = np.mean(train_loss_list)
            # val_loss = np.mean(test_loss_list)
            # print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
            diz_loss['train_loss'].append(train_loss)
            diz_loss['val_loss'].append(val_loss)

        torch.save(self.model, os.path.join(self.model_dir, self.task_name + '_encoder.pt'))

        np.save(os.path.join(self.model_dir, self.task_name + '_train_loss'),np.array(diz_loss['train_loss']))
        np.save(os.path.join(self.model_dir, self.task_name + '_val_loss'),np.array(diz_loss['val_loss']))

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
    def test(self):
        self.visualize()

        # num_epochs = 1
        # diz_loss = {'train_loss': [], 'val_loss': []}
        # for epoch in range(num_epochs):
        #     for i in range(0,20):
        #         self.prepare_dataset(i*10,(i+1)*10)
        #         val_loss = self.test_epoch()
        #         print('\n EPOCH {}/{} \t Index {}-{} \t val loss {}'.format(epoch + 1, num_epochs,i*10,(i+1)*10, val_loss))


    def visualize(self):
        tac_list = []
        tactile_pos_list = []
        object_pos_list = []
        object_pos_pre_list = []

        for i in range(1, 5):
            index = i * 100
            with open(self.save_dir + self.object_name + '_%d.pkl' % index, 'rb') as f:
                data = pickle.load(f)
            tactile_log = np.array(data['tactile']).reshape(-1, 653)
            tactile_pos_log = np.array(data['tac_pose']).reshape(-1, 653, 3)
            object_pos_log = np.array(data['object_pos']).reshape(-1, 6)
            object_pos_pre_log = np.array(data['object_pos_pre']).reshape(-1,6)

            for j in range(tactile_log.shape[0]):
                tactile = tactile_log[j]
                object_pos = object_pos_log[j]
                tactile_pos = tactile_pos_log[j]
                object_pos_pre = object_pos_pre_log[j]

                if tactile[tactile > 0].shape[0] > 5:
                    tac_list.append(tactile)
                    object_pos_list.append(object_pos)
                    tactile_pos_list.append(tactile_pos)
                    object_pos_pre_list.append(object_pos_pre)

        tac = torch.tensor(np.array(tac_list), device=self.device, dtype=torch.float32)
        pos = torch.tensor(np.array(tactile_pos_list), device=self.device, dtype=torch.float32) * 100
        y = torch.tensor(np.array(object_pos_list), device=self.device, dtype=torch.float32) * 100
        object_pos_p = torch.tensor(np.array(object_pos_pre_list), device=self.device, dtype=torch.float32) * 100

        # tac /= tac.max(1,keepdim=True)[0]
        batch_size = 32
        tac = tac.view(-1, 653, 1)

        tactile_dataset = []
        for i in range(tac.shape[0]):
            data = Data(x=tac[i, tac[i, :, 0] != 0, :], pos=pos[i, tac[i, :, 0] != 0, :], y=y[i].view(1, -1), pos_pre = object_pos_p[i,:].view(1,-1))
            tactile_dataset.append(data)

        # m = len(tactile_dataset)
        # tactile_dataset, val_data = random_split(tactile_dataset, [int(m * 0.8), m - int(m * 0.8)])
        self.visua_loader = DataLoader(tactile_dataset, batch_size=batch_size)

        self.model.eval()

        object_pos_pre = []

        if self.model_type == 'gnn' or self.model_type == 'gnn_pre' or self.model_type == 'gcn':

            for data in self.visua_loader:
                logits = self.model(data.x, data.pos, data.batch, data.pos_pre)  # Forward pass.
                logits = logits.cpu().detach().tolist()
                for i in range(len(logits)):
                    object_pos_pre.append(logits[i])

        self.tactile_plot_sim(tac,pos,y,object_pos_pre,object_pos_p)

    def tactile_plot_sim(self,tactile,tactile_pos,object_pos,object_pos_pre,object_pos_p):

        tactile_log = np.array(tactile.cpu())[:,:,0]
        tactile_pos_log = np.array(tactile_pos.cpu())/100
        object_pre_log = np.array(object_pos_pre)/100
        object_pos_log = np.array(object_pos.cpu().detach())/100
        object_pos_p_log = np.array(object_pos_p.cpu())/100

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')

        for i in range(1000,tactile_log.shape[0]):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            i = i
            print(i)
            x = tactile_pos_log[i, :, 0]
            y = tactile_pos_log[i, :, 1]
            z = tactile_pos_log[i, :, 2]
            tac = tactile_log[i]

            ax.clear()
            ax.scatter(x, y, z, s=(tac) * 100 + 1)
            ax.scatter(x, y, z, c='r', s=(tac) * 100)
            ax.scatter(object_pre_log[i, 0], object_pre_log[i, 1], object_pre_log[i, 2], c='g', s=6000)
            ax.scatter(object_pre_log[i, 3], object_pre_log[i, 4], object_pre_log[i, 5], c='g', s=6000)
            ax.scatter(object_pos_log[i, 0], object_pos_log[i, 1], object_pos_log[i, 2], c='orange', s=6000)
            ax.scatter(object_pos_log[i, 3], object_pos_log[i, 4], object_pos_log[i, 5], c='orange', s=6000)
            ax.scatter(object_pos_p_log[i, 0], object_pos_p_log[i, 1], object_pos_p_log[i, 2], c='blue', s=6000)
            ax.scatter(object_pos_p_log[i, 3], object_pos_p_log[i, 4], object_pos_p_log[i, 5], c='blue', s=6000)
            ax.view_init(elev=45, azim=45)
            ax.set(xlim=[-0.1, 0.1], ylim=[-0.1, 0.1], zlim=[0.5, 0.6])

            plt.show()

    def tactile_process(self,data):
        tactile_list = []
        for j in range(len(data)):

            tactile = data[j]
            tac = np.zeros([36,36])

            tac[24,20:20+15] = tactile[:15]
            tac[25,20:20+15] = tactile[15:30]
            tac[26,20:20+15] = tactile[30:45]
            tac[27,20:20+15] = tactile[45:60]

            tac[28,21:21+13] = tactile[60:73]
            tac[29,20+8:20+8+6] = tactile[73:79]
            tac[30,20+8:20+8+6] = tactile[79:85]
            tac[31,20+8:20+8+6] = tactile[85:91]
            tac[32,20+8:20+8+6] = tactile[91:97]

            tac[33,20+8:20+8+5] = tactile[97:102]
            tac[34,20+8:20+8+4] = tactile[102:106]
            tac[35,16+8:16+8+7] = tactile[106:113]

            for i in range(12):
                tac[12-(i+1),18:18+6] = tactile[113+i*6:113+(i+1)*6]
            for i in range(12):
                tac[12+i,18:18+6] = tactile[113+72+i*6:113+72+(i+1)*6]

            for i in range(12):
                tac[12-(i+1),24:24+6] = tactile[113+144+i*6:113+144+(i+1)*6]
            for i in range(12):
                tac[12+i,24:24+6] = tactile[113+144+72+i*6:113+144+72+(i+1)*6]

            for i in range(12):
                tac[12-(i+1),30:36] = tactile[113+288+i*6:113+288+(i+1)*6]
            for i in range(12):
                tac[12+i,30:36] = tactile[113+288+72+i*6:113+288+72+(i+1)*6]

            for i in range(12):
                tac[-6:,12-(i+1)] = tactile[113+432+i*6:113+432+(i+1)*6]
            for i in range(6):
                tac[-6:,12+i] = tactile[113+432+72+i*6:113+432+72+(i+1)*6]

            tac_clone = tac.copy()
            for i in range(36):
                tac[:,i] = tac_clone[:,35-i]
            tactile_list.append(tac)

            # fig = plt.figure(figsize=(8, 8))
            # plt.imshow(tac)
            # plt.colorbar()
            # plt.show()
        return np.array(tactile_list)

if __name__ == "__main__":

    ENCODER = encoder()
    ENCODER.run()