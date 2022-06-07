import os.path
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader,random_split,TensorDataset
import pickle
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
from model import *

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './models'
if_model = False
save_dir = '../tac_data/'

task_name = 'cylinder_big'
object_name = 'cylinder_small'
# object_name_2 = '010_potted_meat_can'
# object_name_3 = '025_mug'
# object_name_4 = '061_foam_brick'

with open(save_dir+object_name+'.pkl','rb') as f:
    d = pickle.load(f)
tac = abs(torch.tensor(d['tactile'], device=device)[:,:,2])
y = torch.tensor(d['class'], device=device)

# with open(save_dir+object_name_2+'.pkl','rb') as f:
#     d_2 = pickle.load(f)
# with open(save_dir+object_name_3+'.pkl','rb') as f:
#     d_3 = pickle.load(f)
# with open(save_dir+object_name_4+'.pkl','rb') as f:
#     d_4 = pickle.load(f)

# tac = abs(torch.tensor(np.concatenate((d['tactile'],d_2['tactile'],d_3['tactile'],d_4['tactile']),axis=0), device=device)[:,:,2])
# y = torch.tensor(np.concatenate((d['class'],d_2['class'],d_3['class'],d_4['class']),axis=0), device=device)

tac /= tac.max(1,keepdim=True)[0]

x = tac.reshape(-1,1,15,15)
y = y
tactile_dataset = TensorDataset(x,y)

m=len(tactile_dataset)

train_data, val_data = random_split(tactile_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 10

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
if if_model:
    encoder = torch.load(os.path.join(model_dir,task_name+'_encoder.pt'))
    decoder = torch.load(os.path.join(model_dir,task_name+'_decoder.pt'))
else:
    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    t_idx = np.random.randint(0,val_data.__len__(),n)
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = val_data[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap=plt.cm.hot_r)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap=plt.cm.hot_r)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

if not if_model:
    num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
       train_loss =train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
       val_loss = test_epoch(encoder,decoder,device,valid_loader,loss_fn)
       print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
       diz_loss['train_loss'].append(train_loss)
       diz_loss['val_loss'].append(val_loss)

    torch.save(encoder, os.path.join(model_dir, task_name + '_encoder.pt'))
    torch.save(decoder, os.path.join(model_dir, task_name + '_decoder.pt'))

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

plot_ae_outputs(encoder, decoder, n=10)

encoded_samples = []
import tqdm
for sample in val_data.dataset:
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    # Encode image
    encoder.eval()
    with torch.no_grad():
        encoded_img  = encoder(img)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)

import pandas as pd
encoded_samples = pd.DataFrame(encoded_samples)

import plotly.express as px

fig= px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1',
           color=encoded_samples.label.astype(str), opacity=0.7)
fig.show()

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
fig = px.scatter(tsne_results, x=0, y=1,
                 color=encoded_samples.label.astype(str),
                 labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
fig.show()