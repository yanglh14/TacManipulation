import os.path
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
from torch.utils.data import DataLoader,random_split,TensorDataset
import pickle
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import torch
from torchvision import transforms
from model import *
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './models'
if_model = True
save_dir = '../tac_data/'

task_name = 'final_2'
object_name = 'tac_data'
# object_name_2 = '010_potted_meat_can'
# object_name_3 = '025_mug'
# object_name_4 = '061_foam_brick'

with open(save_dir+object_name+'.pkl','rb') as f:
    d = pickle.load(f)
tac = abs(torch.tensor(d['tactile'],dtype=torch.float32, device=device)[:,:,2])
y = torch.tensor(d['class'],dtype=torch.long, device=device)

# with open(save_dir+object_name+'.pkl','rb') as f:
#     d = pickle.load(f)
# with open(save_dir+object_name_2+'.pkl','rb') as f:
#     d_2 = pickle.load(f)
# with open(save_dir+object_name_3+'.pkl','rb') as f:
#     d_3 = pickle.load(f)
# with open(save_dir+object_name_4+'.pkl','rb') as f:
#     d_4 = pickle.load(f)

# tac = abs(torch.tensor(np.concatenate((d['tactile'],d_2['tactile'],d_3['tactile'],d_4['tactile']),axis=0),dtype=torch.float32, device=device)[:,:,2])
# y = torch.tensor(np.concatenate((d['class'],d_2['class'],d_3['class'],d_4['class']),axis=0),dtype=torch.long, device=device)

tac /= tac.max(1,keepdim=True)[0]

x = tac.reshape(-1,1,15,15)
y = y
tactile_dataset = TensorDataset(x,y)

m=len(tactile_dataset)

train_data, val_data = random_split(tactile_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=32

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size)

### Define the loss function
loss_fn = nn.CrossEntropyLoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 10

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
if if_model:
    classifier = torch.load(os.path.join(model_dir,task_name+'_classifier.pt'))
else:
    classifier = classifier(encoded_space_dim=d)

params_to_optimize = [
    {'params': classifier.parameters()},
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Move both the encoder and the decoder to the selected device
classifier.to(device)

### Training function
def train_epoch(classifier, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    classifier.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, labels in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = classifier(image_batch)

        # Evaluate loss
        loss = loss_fn(encoded_data, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(classifier, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, labels in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = classifier(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(encoded_data.cpu())
            conc_label.append(labels.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def accuracy_epoch(classifier, device, dataloader,one_class=None):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        correct = 0
        total = 0

        for image_batch, labels in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            labels = labels.to(device)

            # Encode data
            if one_class != None:
                image_batch = image_batch[labels==one_class]
                labels = labels[labels==one_class]

            if labels.size(0)>0:
                encoded_data = classifier(image_batch)
                _, predicted = torch.max(encoded_data.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    Accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {Accuracy} %')

    return Accuracy

def probablity_dis(classifier, device, dataloader,one_class=None):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad(): # No need to track the gradients
        First = True
        for image_batch, labels in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            labels = labels.to(device)

            # Encode data
            if one_class != None:
                image_batch = image_batch[labels==one_class]
                labels = labels[labels==one_class]

            if labels.size(0)>0:
                encoded_data = classifier(image_batch)

                if First:
                    predicted_list = encoded_data.cpu().numpy()
                    First = False
                else:
                    predicted_list = np.append(predicted_list,encoded_data.cpu().numpy(),axis=0)
        probablity = np.average(predicted_list,axis=0)

        return probablity

if not if_model:
    num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}
    Accuracy_list = []
    for epoch in range(num_epochs):
       train_loss = train_epoch(classifier,device,train_loader,loss_fn,optim)
       val_loss = test_epoch(classifier,device,valid_loader,loss_fn)
       print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
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
    # accuracy_list = []
    # for i in range(10):
    #     accuracy = accuracy_epoch(classifier, device, valid_loader,one_class=i)
    #     accuracy_list.append(accuracy)
    #
    # fig, ax = plt.subplots()
    # labels = list(range(10))
    # ax.bar(labels, accuracy_list, 0.35, label='Test')
    #
    # ax.set_xlabel('Object Class')
    # ax.set_ylabel('Average Accuracy')
    # ax.legend()
    # plt.show()

    pro_list = []
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

    # encoded_samples = []
    #
    # for sample in val_data.dataset:
    #     img = sample[0].unsqueeze(0).to(device)
    #     label = sample[1]
    #     # Encode image
    #     classifier.eval()
    #     with torch.no_grad():
    #         encoded_img = classifier(img)
    #     # Append to list
    #     encoded_img = encoded_img.flatten().cpu().numpy()
    #     encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    #     encoded_sample['label'] = label
    #     encoded_samples.append(encoded_sample)
    #
    # encoded_samples = pd.DataFrame(encoded_samples)
    #
    # # fig= px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1',
    # #            color=encoded_samples.label.astype(str), opacity=0.7)
    # # fig.show()
    #
    # tsne = TSNE(n_components=2)
    # tsne_results = tsne.fit_transform(encoded_samples.drop(['label'], axis=1))
    # fig = px.scatter(tsne_results, x=0, y=1,
    #                  color=encoded_samples.label.astype(str),
    #                  labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    # fig.show()