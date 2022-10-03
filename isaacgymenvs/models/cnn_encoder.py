import os.path
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader,random_split,TensorDataset
import pickle
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
from model import *

def tactile_process(data):
    tactile_list = []
    for i in range(len(data)):

        tactile = data[i]
        tac = np.zeros([36,36])

        tac[24,18:18+15] = tactile[:15]
        tac[25,18:18+15] = tactile[15:30]
        tac[26,18:18+15] = tactile[30:45]
        tac[27,18:18+15] = tactile[45:60]

        tac[28,19:19+13] = tactile[60:73]
        tac[29,18+8:18+8+6] = tactile[73:79]
        tac[30,18+8:18+8+6] = tactile[79:85]
        tac[31,18+8:18+8+6] = tactile[85:91]
        tac[32,18+8:18+8+6] = tactile[91:97]

        tac[33,18+8:18+8+5] = tactile[97:102]
        tac[34,18+8:18+8+4] = tactile[102:106]
        tac[35,18+8:18+8+7] = tactile[106:113]

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
        tactile_list.append(tac)
    return np.array(tactile_list)
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './checkpoint'
if_model = False
save_dir = '../runs_tac/'

task_name = 'ball_cnn'
object_name = 'dataset'


plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 30

tac_list = []
object_pos_list = []

tac = np.zeros([1,653])
tac[0,:113] = 1
tac[0,113:113+144] = 0.8
tac[0,113+144:113+144+144] = 0.3
tac[0,113+144+144:113+144+144+144] = 0.8
tac[0,-108:] = 0.5

tac = tactile_process(tac)[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(tac,cmap=plt.cm.gray)
plt.show()
for i in range (1,50):
    index = i *100
    with open(save_dir+object_name+'_%d.pkl'%index,'rb') as f:
        data = pickle.load(f)
    tactile_log = np.array(data['tactile']).reshape(-1,653)
    tactile_pos_log = np.array(data['tac_pose']).reshape(-1,653,3)
    object_pos_log = np.array(data['object_pos']).reshape(-1,6)
    for j in range(tactile_log.shape[0]):
        tactile = tactile_log[j]
        object_pos = object_pos_log[j]
        if tactile[tactile>0].shape[0] >5:
            tac_list.append(tactile)
            object_pos_list.append(object_pos)

tac = tactile_process(tac_list)

tac = torch.tensor(np.array(tac), device=device, dtype=torch.float32)
y = torch.tensor(np.array(object_pos_list), device=device, dtype=torch.float32)

# tac /= tac.max(1,keepdim=True)[0]

x = tac.reshape(-1,1,36,36)
y = y * 100
tactile_dataset = TensorDataset(x,y)

m=len(tactile_dataset)

train_data, val_data = random_split(tactile_dataset, [int(m*0.8), m - int(m*0.8)])
batch_size=32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)


#model = Autoencoder(encoded_space_dim=encoded_space_dim)
if if_model:
    encoder = torch.load(os.path.join(model_dir,task_name+'_encoder.pt'))
else:
    encoder = CNNEncoder(encoded_space_dim=6)

params_to_optimize = [
    {'params': encoder.parameters()},
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


# Move both the encoder and the decoder to the selected device
encoder.to(device)

### Training function
def train_epoch(encoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, labels in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)

        # Evaluate loss
        loss = loss_fn(encoded_data, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(encoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, labels in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(encoded_data.cpu())
            conc_label.append(labels.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

if not if_model:
    num_epochs = 100
    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
       train_loss = train_epoch(encoder,device,train_loader,loss_fn,optim)
       val_loss = test_epoch(encoder,device,valid_loader,loss_fn)
       print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
       diz_loss['train_loss'].append(train_loss)
       diz_loss['val_loss'].append(val_loss)

    torch.save(encoder, os.path.join(model_dir, task_name + '_encoder.pt'))

    np.save(os.path.join(model_dir, task_name + '_train_loss'),np.array(diz_loss['train_loss']))
    np.save(os.path.join(model_dir, task_name + '_val_loss'),np.array(diz_loss['val_loss']))

if if_model:

    # diz_loss = {'train_loss': [], 'val_loss': []}
    # diz_loss['train_loss'] = np.load('./data/final_2_train_loss.npy')
    # diz_loss['val_loss'] = np.load('./data/final_2_val_loss.npy')
    #
    # # Plot losses
    # plt.figure(figsize=(10,8))
    # plt.semilogy(diz_loss['train_loss'], label='Train')
    # plt.semilogy(diz_loss['val_loss'], label='Valid')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Average Loss')
    # plt.ylim([0,0.1])
    # #plt.grid()
    # plt.legend()
    # plt.show()

    # plot_ae_outputs(encoder, decoder, n=10)

    encoded_samples = []

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

    # fig= px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1',
    #            color=encoded_samples.label.astype(str), opacity=0.7)
    # fig.show()

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
    fig = px.scatter(tsne_results, x=0, y=1,
                     color=encoded_samples.label.astype(str),
                     labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})

    fig.show()

    # label = np.array(encoded_samples.label.astype(float))
    # fig, ax = plt.subplots()
    #
    # scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1],c = label)
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Classes")
    # ax.add_artist(legend1)
    # plt.show()