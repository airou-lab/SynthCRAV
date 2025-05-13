#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import time
import struct
from copy import copy

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchviz
import wandb

from nuscenes import NuScenes
from nuscenes.utils import splits

# from models.models_utils.utils import *
# from models.models_utils.config import device, ndevice
# from models.models import RadarNDet, CameraNDet

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

if torch.cuda.is_available():
    device = torch.device('cuda')
    ndevice = torch.cuda.current_device()
    # if os.getpid() == 1:  # Not working on windows distro
    #     print('\nfound CUDA device:', torch.cuda.get_device_name(ndevice))
    print('\nfound CUDA device:', torch.cuda.get_device_name(ndevice))
else:
    # if os.getpid() == 1: # Not working on windows distro
    #     print('\nno CUDA installation found, using CPU')
    print('\nno CUDA installation found, using CPU')
    device = torch.device('cpu')
    ndevice = torch.cuda.current_device()

# UTILS
def decode_pcd_file(filename,verbose=False):
    # Extract sensor data
    if verbose:
        print('Opening point cloud data at:', filename)

    # opening file    
    meta = []
    with open (filename, 'rb') as file:
        for line in file:
            line = line.strip().decode('utf-8')
            meta.append(line)                        

            if line.startswith('DATA'):
                break

        data_binary = file.read()

    #extracting headers
    fields = meta[2].split(' ')[1:]
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)                    
    
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
             'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
             'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # store in dataframe
    df = pd.DataFrame(points,columns=fields, dtype=object)

    return df, types_str

def convert_radardf_to_tensor(radar_df, types_str):
    npdtype_list = []
    tensor_list = []
    torchdtype = torch.float32

    for typechar in types_str:
        # floats
        if typechar == 'e':
            npdtype = np.float16

        elif typechar == 'f':
            npdtype = np.float32

        elif typechar == 'd':
            npdtype = np.float64
            torchdtype = torch.float64 # promote to 64 floats if at least one column is in this type

        # signed int
        elif typechar == 'b':
            npdtype = np.int8

        elif typechar == 'h':
            npdtype = np.int16

        elif typechar == 'i':
            npdtype = np.int32

        elif typechar == 'q':
            npdtype = np.int64

        # unsigned int
        elif typechar == 'B':
            npdtype = np.uint8

        elif typechar == 'H':
            npdtype = np.uint16

        elif typechar == 'I':
            npdtype = np.uint32

        elif typechar == 'Q':
            npdtype = np.uint64

        npdtype_list.append(npdtype)

    for col, dtypenp in zip(radar_df.columns,npdtype_list):
        tensor_list.append(torch.tensor(radar_df[col].values.astype(dtypenp), dtype=torch.float32))
        
    combined_tensor = torch.stack(tensor_list, dim=-1)

    return combined_tensor

def get_labels(data_module):
    # Loading labels
    labels = copy(data_module.df_train['labels']).drop_duplicates().sort_values().reset_index(drop=True)
    n_labels = len(labels)

    return labels, len(labels)

# Dataset loading utils
def get_df_split(nusc, args, sensor_type, data_split):
    '''
    For Cameras the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<noise_type>/<name.jpg>
    For Radars the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<name.pcd>
    '''
    # output
    data_paths = []
    labels = []
    sensors_list = []

    # accumulate in df:
    for scene in nusc.scene:
        if scene['name'] not in data_split:
            continue

        nusc_sample = nusc.get('sample', scene['first_sample_token'])

        while True:
            if sensor_type == 'CAM':
                for sensor in cam_list:
                    # Load nusc info
                    sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                    filename = sample_data['filename']
                    token = filename.split('/')[-1]

                    getOG=False

                    for noise_level in range (10,110,10):
                        for noise_type in ['Blur', 'Gaussian_noise', 'High_exposure', 'Low_exposure']:
                            synthpath = os.path.join(args.data_root,'samples',sensor,str(noise_level),noise_type,token)
                            if os.path.exists(synthpath):
                                data_paths.append(synthpath)
                                labels.append(int(noise_level/10))
                                sensors_list.append(sensor)
                                getOG=True # signal flag that data is good to take from OG as well

                if getOG:
                    data_paths.append(os.path.join(args.nusc_root,filename))
                    labels.append(0)
                    sensors_list.append(sensor)

            elif sensor_type == 'RADAR':
                for sensor in radar_list:
                    # Load nusc info
                    sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                    filename = sample_data['filename']
                    token = filename.split('/')[-1]

                    getOG=False

                    for noise_level in range (10,110,10):
                        synthpath = os.path.join(args.data_root,'samples',sensor,str(noise_level),token)
                        if os.path.exists(synthpath):
                            # removing empty dataframes
                            radar_df, types_str = decode_pcd_file(synthpath,verbose=False)
                            if not radar_df.isnull().values.any():
                                data_paths.append(synthpath)
                                labels.append(int(noise_level/10))
                                sensors_list.append(sensor)
                            getOG=True # signal flag that data is good to take from OG as well


                    if getOG:
                        # removing empty dataframes
                        radar_df, types_str = decode_pcd_file(synthpath,verbose=False)
                        if not radar_df.isnull().values.any():
                            data_paths.append(os.path.join(args.nusc_root,filename))
                            labels.append(0)
                            sensors_list.append(sensor)


            if nusc_sample['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)
    
    df = pd.DataFrame({'data':data_paths,'labels':labels,'sensor':sensors_list})

    return df

def create_df(args, nusc, sensor):
    # accumulate scene names
    trainval = splits.mini_train
    test_split = splits.mini_val

    if sensor == 'CAM':
        trainval = trainval[:-3]  # removing night scenes for camera noise

    n_train_scenes = int(len(trainval)*args.ntrain)
    n_val_scenes = len(trainval) - n_train_scenes


    train_split, val_split = random_split(trainval,[n_train_scenes,n_val_scenes],generator=torch.Generator().manual_seed(42))

    if args.smaller_dataset:
        # Smaller dataset with only 1 scenes for train/val/test, totalling 3 scenes. Required for smaller configs
        n_train_scenes = 1
        n_val_scenes = 1
        train_split = random.choice(list(train_split))
        val_split = random.choice(list(val_split))
        test_split = random.choice(list(test_split))
    else:
        # using the full dataset, requires high-end computer
        train_split = list(train_split)
        val_split = list(val_split)
        test_split = list(test_split)

    # Generating output splits
    
    df_train = get_df_split(nusc, args, sensor, train_split)
    df_val   = get_df_split(nusc, args, sensor, val_split)
    df_test  = get_df_split(nusc, args, sensor, test_split)

    if args.verbose:
        print('sensor:',sensor)
        print('trainval:',trainval)
        print('n_train_scenes:',n_train_scenes)
        print('n_val_scenes:',n_val_scenes)
        
        print('train_split:',train_split)
        print('val_split:',val_split)
        print('test_split:',test_split)

        print('train dataset:',df_train)
        print('test dataset:',df_val)
        print('val dataset:',df_test)

    return df_train, df_val, df_test  

def load_pcd(row):
    '''
    takes a df in input (batch)
    returns a tensor with the loaded images
    '''
    labels = torch.tensor(row['labels'], dtype=torch.long)
    radar_df, types_str = decode_pcd_file(row['data'],verbose=False)

    data = convert_radardf_to_tensor(radar_df,types_str)
    # print (data.shape)
    # data = data.unsqueeze(0).transpose(1, 2)  # Shape: [1, N, C] -> [1, C, N]
    data = data.transpose(0, 1)  # Shape: [N, C] -> [C, N]

    return data, labels, radar_df

def load_pcd_masked(row):
    '''
    takes a df in input (batch)
    returns a tensor with the loaded images
    '''
    N=256
    labels = torch.tensor(row['labels'], dtype=torch.long)
    radar_df, types_str = decode_pcd_file(row['data'],verbose=False)

    data = convert_radardf_to_tensor(radar_df,types_str)

    n_pads = N - data.shape[0]

    if n_pads>0:
        pad = torch.zeros((n_pads, data.shape[1]), dtype=data.dtype, device=data.device)
        padded_data = torch.cat([data, pad], dim=0)
    else:
        padded_data = data[:N]


    # print('data.shape:',data.shape)
    # data = data.unsqueeze(0).transpose(1, 2)  # Shape: [1, N, C] -> [1, C, N]
    padded_data = padded_data.transpose(0, 1)  # Shape: [N, C] -> [C, N]

    # print('padded_data:',padded_data)
    # print('padded_data.shape:',padded_data.shape)
    # input()

    # Creating a mask to ignore padded points in training
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:min(data.shape[0], N)] = True

    return padded_data, labels, mask

def load_imgs(row):
    '''
    takes a row of a df in input
    returns a tensor with the loaded image and a tensor of its label
    '''
    img = cv2.imread(row['data'])

    if img is None:
        raise ValueError(f"Image not found at {row['data']}")

    data = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    labels = torch.tensor(row['labels'], dtype=torch.long)
    return data, labels, img

# DATALOADER
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data, labels, _ =load_imgs(row)
        return data, labels

class RadarDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # data, labels, _ =load_pcd(row)
        data, labels, mask =load_pcd_masked(row)
        if not data.shape[1]:  # Skip empty point clouds
            return None
        return data, labels, mask

    def collate_fn(batch):
        xs, ys, dfs, masks = zip(*batch)
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        masks = torch.stack(masks)
        return xs, ys, masks


class DataModule(pl.LightningDataModule):
    def __init__(self, args, sensor, batch_size=16, n_workers=0):
        super().__init__()

        nusc = NuScenes(version='v1.0-mini', dataroot=args.nusc_root, verbose=True)# loading nusc table
        self.df_train, self.df_val, self.df_test = create_df(args, nusc, sensor)
        self.batch_size = batch_size
        self.sensor = sensor
        self.n_workers = n_workers


    def train_dataloader(self):
        if self.sensor == 'CAM':
            train_dataset = ImageDataset(self.df_train)
        elif self.sensor == 'RADAR':
            train_dataset = RadarDataset(self.df_train)

        if self.n_workers:
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, persistent_workers=True,  pin_memory=True)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.sensor == 'CAM':
            val_dataset = ImageDataset(self.df_val)
        elif self.sensor == 'RADAR':
            val_dataset = RadarDataset(self.df_val)
        if self.n_workers:
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, persistent_workers=True,  pin_memory=True)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if self.sensor == 'CAM':
            test_dataset = ImageDataset(self.df_test)
        elif self.sensor == 'RADAR':
            test_dataset = RadarDataset(self.df_test)
        if self.n_workers:
            return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, persistent_workers=True,  pin_memory=True)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    

# Visualization utils
# Neural Net plots
def plot_trainval_loss(hist_filename):
    plt.figure(figsize=(16, 9))

    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    train_loss = hist['train_loss']
    val_loss = hist['val_loss']
    n_epochs = len(train_loss)

    print()
    print('n_epochs:',n_epochs)
    print('train_loss:',train_loss)
    print('val_loss:',val_loss)

    plt.plot(range(n_epochs),train_loss)
    plt.plot(range(n_epochs),val_loss)

    plt.title('Train/Val loss vs Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend(['Train loss','Val loss'])

    plt.show()



def plot_trainval_acc(hist_filename):
    plt.figure(figsize=(16, 9))

    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    train_acc = hist['train_accuracy']
    val_acc = hist['val_accuracy']
    n_epochs = len(train_acc)

    print()
    print('n_epochs:',n_epochs)
    print('train_acc:',train_acc)
    print('val_acc:',val_acc)

    plt.plot(range(n_epochs),train_acc)
    plt.plot(range(n_epochs),val_acc)

    plt.title('Train/Val accuracy vs Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend(['Train accuracy','Val accuracy'])

    plt.show()


def get_test_acc(hist_filename):
    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    test_loss = hist['test_loss']
    test_accuracy = hist['test_accuracy']

    print()
    print('test_loss:',test_loss)
    print('test_accuracy:',test_accuracy)

def get_TP_FP(hist_filename):
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f)

    preds = hist['test_results']['preds']
    labels = hist['test_results']['labels']
    n_classes = 11
    TP = np.zeros(n_classes, dtype=int)
    FN = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        TP[c] = np.sum((preds == c) & (labels == c))  # Correctly predicted class c
        FN[c] = np.sum((preds != c) & (labels == c))  # Missed class c

    print('TP:',TP,sum(TP))
    print('FN:',FN,sum(FN))

def plot_confusion_mat(y_true, y_pred, name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print('TP:',np.trace(cm))
    print('FN:',len(y_true)-np.trace(cm))

    num_labels = np.arange(0,11,dtype=int).tolist()
    label_list = np.array([0,10,20,30,40,50,60,70,80,90,100])

    fig, ax = plt.subplots(figsize=(16,9))
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, labels=num_labels, display_labels=label_list, ax=ax, colorbar=False)
    disp.plot(ax=ax,cmap=plt.cm.Blues, xticks_rotation=45)
    plt.savefig(name,dpi=300)





# Radar noise classifier
class RadarNDet(pl.LightningModule):
    def __init__(self, n_channels, output_size, history, dropout_prob=0, lr=1e-3):
        super(RadarNDet, self).__init__()

        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history

        # ------------------------------------------model------------------------------------------
        # self.input =  nn.Linear(input_size, embed_size)
        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=9, padding=4),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    )

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    )

        self.fc1 =  nn.Sequential(nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob)
                                    )

        self.attention1 = nn.MultiheadAttention(32, num_heads=1, batch_first=True)

        self.head = nn.Sequential(nn.Linear(32, output_size))
        # no softmax because we use cross entropy loss

    def forward(self, x,mask=None):
        x = self.input(x)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        x = self.fc1(x)

        if mask is not None:
            # mask should be (B, N), False = padding, True = keep
            x, _ = self.attention1(x, x, x, key_padding_mask=~mask )
        else:
            x, _ = self.attention1(x, x, x)

        # x, _ = self.attention1(x, x, x)
        last_hidden = x[:, -1, :]
        x = self.head(last_hidden)

        return x
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-20,weight_decay=1e-5)



    # train 
    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_hat = self(x, mask=mask)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1) 
        acc = (preds == y).float().mean()


        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, prog_bar=True) 
        # self.log('train_acc_epoch', acc, prog_bar=False, on_step=False, on_epoch=True)

        self.train_step_outputs['loss'].append(loss.item())
        self.train_step_outputs['acc'].append(acc)

        return loss

    def on_train_epoch_end(self):
        avg_acc = torch.stack(self.train_step_outputs['acc']).mean()
        tot_loss = sum(self.train_step_outputs['loss'])
        self.log('train_acc', avg_acc, prog_bar=False)

        self.history['train_loss'].append(tot_loss)
        self.history['train_accuracy'].append(avg_acc)

        self.train_step_outputs['acc']=[]  # clear for next epoch
        self.train_step_outputs['loss']=[]  # clear for next epoch

        wandb.log({'loss': tot_loss, 'accuracy': avg_acc}, step=self.current_epoch)



    # val
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_hat = self(x, mask=mask)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        if not self.trainer.sanity_checking:
            self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
            self.log('val_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

            # self.log('val_loss', loss, prog_bar=True)
            # self.log('val_acc', acc, prog_bar=True)

            self.validation_step_outputs['loss'].append(loss)
            self.validation_step_outputs['acc'].append(acc)
        

        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            avg_acc = torch.stack(self.validation_step_outputs['acc']).mean()
            tot_loss = sum(self.validation_step_outputs['loss'])
            self.log('val_acc', avg_acc, prog_bar=False)

            self.history['val_loss'].append(tot_loss)
            self.history['val_accuracy'].append(avg_acc)

            self.validation_step_outputs['acc']=[]  # clear for next epoch
            self.validation_step_outputs['loss']=[]  # clear for next epoch
        
            wandb.log({'val_loss': tot_loss, 'val_accuracy': avg_acc}, step=self.current_epoch)


    # test
    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_hat = self(x, mask=mask)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('test_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('test_acc', acc, prog_bar=True)

        self.test_step_outputs['loss'].append(loss.item())
        self.test_step_outputs['acc'].append(acc)
        self.test_step_outputs['preds'].append(preds.cpu())
        self.test_step_outputs['labels'].append(y.cpu())

        return loss    

    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_step_outputs['acc']).mean()
        tot_loss = sum(self.test_step_outputs['loss'])
        self.log('test_acc', avg_acc, prog_bar=False)

        self.history['test_loss'].append(tot_loss)
        self.history['test_accuracy'].append(avg_acc)

        all_preds = torch.cat(self.test_step_outputs['preds']).numpy()
        all_labels = torch.cat(self.test_step_outputs['labels']).numpy()

        plot_confusion_mat(all_preds, all_labels, os.path.join('./ckpt','radar_model_mat.png'))

        wandb.log({'test_loss': tot_loss, 'test_accuracy': avg_acc}, step=self.current_epoch)

# Camera noise classifier
class CameraNDet(pl.LightningModule):
    def __init__(self, image_shape, output_size, history, conv_k=3, dropout_prob=0, lr=1e-3):
        super(CameraNDet, self).__init__()
        
        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history

        image_size = np.array(image_shape[:2]) # 1080 x 1920
        in_channels = image_shape[2] # 3

        # ------------------------------------------model------------------------------------------
        self.input = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) 

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) 
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #


        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #

        self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(8, output_size)
                    )

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.head(x)
        return x
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-20)



    # train 
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1) 
        acc = (preds == y).float().mean()


        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, prog_bar=True) 
        # self.log('train_acc_epoch', acc, prog_bar=False, on_step=False, on_epoch=True)

        self.train_step_outputs['loss'].append(loss.item())
        self.train_step_outputs['acc'].append(acc)

        return loss

    def on_train_epoch_end(self):
        avg_acc = torch.stack(self.train_step_outputs['acc']).mean()
        tot_loss = sum(self.train_step_outputs['loss'])
        self.log('train_acc', avg_acc, prog_bar=False)

        self.history['train_loss'].append(tot_loss)
        self.history['train_accuracy'].append(avg_acc)

        self.train_step_outputs['acc']=[]  # clear for next epoch
        self.train_step_outputs['loss']=[]  # clear for next epoch

        wandb.log({'loss': tot_loss, 'accuracy': avg_acc}, step=self.current_epoch)



    # val
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        if not self.trainer.sanity_checking:
            self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
            self.log('val_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

            # self.log('val_loss', loss, prog_bar=True)
            # self.log('val_acc', acc, prog_bar=True)

            self.validation_step_outputs['loss'].append(loss)
            self.validation_step_outputs['acc'].append(acc)
        

        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            avg_acc = torch.stack(self.validation_step_outputs['acc']).mean()
            tot_loss = sum(self.validation_step_outputs['loss'])
            self.log('val_acc', avg_acc, prog_bar=False)

            self.history['val_loss'].append(tot_loss)
            self.history['val_accuracy'].append(avg_acc)

            self.validation_step_outputs['acc']=[]  # clear for next epoch
            self.validation_step_outputs['loss']=[]  # clear for next epoch
        
            wandb.log({'val_loss': tot_loss, 'val_accuracy': avg_acc}, step=self.current_epoch)


    # test
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('test_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('test_acc', acc, prog_bar=True)

        self.test_step_outputs['loss'].append(loss.item())
        self.test_step_outputs['acc'].append(acc)
        self.test_step_outputs['preds'].append(preds.cpu())
        self.test_step_outputs['labels'].append(y.cpu())

        return loss    

    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_step_outputs['acc']).mean()
        tot_loss = sum(self.test_step_outputs['loss'])
        self.log('test_acc', avg_acc, prog_bar=False)

        self.history['test_loss'].append(tot_loss)
        self.history['test_accuracy'].append(avg_acc)

        all_preds = torch.cat(self.test_step_outputs['preds']).numpy()
        all_labels = torch.cat(self.test_step_outputs['labels']).numpy()

        plot_confusion_mat(all_preds, all_labels, os.path.join('./ckpt','camera_model_mat.png'))

        wandb.log({'test_loss': tot_loss, 'test_accuracy': avg_acc}, step=self.current_epoch)



import argparse

def create_parser():

    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument('--nusc_root', type=str, default='./data/default_nuScenes/', help='Original nuScenes data folder')
    parser.add_argument('--data_root', type=str, default='./data/noisy_nuScenes/', help='Synth data folder')
    parser.add_argument('--output_path', type=str, default='./ckpt/', help='checkpoint save path')

    # Network selection
    parser.add_argument('--cameraNR', action='store_true', default=False, help='run CameraNR script')
    parser.add_argument('--radarNR', action='store_true', default=False, help='run RadarNR script')
    parser.add_argument('--smaller_dataset', action='store_true', default=False, help='Use only 1 scene per fold')
    parser.add_argument('--ntrain', type=float, default=0.7, help='Set train set size')

    # misc
    parser.add_argument('--n_workers',type=int, default=0, help='Set a number of workers to load the data' )

    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # actions
    # parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--eval','-e', action='store_true', default=False, help='Evaluate model')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='load checkpoint at <output_path>/<sensor_type>+_model.pth')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model at <output_path>/<sensor_type>+_model.pth')
    # parser.add_argument('--save_hist', action='store_true', default=False, help='save model history at <output_path>/<sensor_type>+_hist.pkl')


    # network parameters
    parser.add_argument('--conv_k', type=int, default=3, help='2D convolution kernel')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

    # Verbosity level
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print(args)

    if args.n_workers>0:
        mp.set_start_method('spawn', force=True)
    
    #-------------------------------------------------------CAMERA-------------------------------------------------------
    if args.cameraNR:
        # Init history memory
        history = history = {'train_loss':[],
                                'val_loss':[],
                                'test_loss':[],
                                'train_accuracy':[],
                                'val_accuracy':[],
                                'test_accuracy':[],
                                'test_results':[]}

        args.lr = 5e-3
        args.n_epochs = 20
        args.batch_size = 32
        args.conv_k = 3
        args.dropout = 0.4

        wandb.init(project='CameraNR',config=vars(args))

        # Load data
        data_module = DataModule(args=args, sensor='CAM',batch_size=args.batch_size,n_workers=args.n_workers)    
        # Loading labels
        labels, n_labels = get_labels(data_module)
        # init model
        model = CameraNDet(image_shape=[900,1600,3], output_size=n_labels, history=history, conv_k=args.conv_k, dropout_prob=args.dropout, lr=args.lr).to(device)
        model_ckpt_filename = 'camera_model.pth'
        hist_filename = 'camera_model_hist.pkl'
        # logger = WandbLogger(project="CameraNR", name='EXP0')


        if args.load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(args.output_path, model_ckpt_filename)))
            with open(os.path.join(args.output_path,hist_filename), 'rb') as f:
                model.history = pickle.load(f)


        # init model
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')
        # trainer = pl.Trainer(logger=logger, max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')

        # training
        trainer.fit(model, data_module)

        start_t = time.perf_counter()
        trainer.test(model, data_module)
        end_t = time.perf_counter()
        print('evaluated %d point clouds in %f seconds'%(len(data_module.df_test),end_t-start_t))

        # save hist (with test output)
        with open(os.path.join(args.output_path, hist_filename), 'wb') as f:
            pickle.dump(history,f)


    #-------------------------------------------------------RADAR-------------------------------------------------------
    if args.radarNR:
        # Init history memory
        history = history = {'train_loss':[],
                                'val_loss':[],
                                'test_loss':[],
                                'train_accuracy':[],
                                'val_accuracy':[],
                                'test_accuracy':[],
                                'test_results':[]}

        args.lr = 1e-3
        args.n_epochs = 250
        args.batch_size = 128
        args.dropout = 0.4

        if args.n_workers>0:
            mp.set_start_method('spawn', force=True)

        # Load data
        data_module = DataModule(args=args, sensor='RADAR',batch_size=args.batch_size,n_workers=args.n_workers)
        # Loading labels
        labels, n_labels = get_labels(data_module)
        # init model
        model = RadarNDet(n_channels=18, output_size=n_labels, history=history, dropout_prob=args.dropout, lr=args.lr).to(device)
        model_ckpt_filename = 'radar_model.pth'
        hist_filename = 'radar_model_hist.pkl'
        # logger = TensorBoardLogger('lightning_logs', name='radar_noise_classifier')
        # logger = WandbLogger(project="RadarNR", name='EXP0')

        batch = next(iter(data_module.train_dataloader()))
        input_shape = batch[0].shape

        # print(batch)
        # print(input_shape)
        # raise KeyboardInterrupt



        if args.load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(args.output_path, model_ckpt_filename)))
            with open(os.path.join(args.output_path,hist_filename), 'rb') as f:
                model.history = pickle.load(f)


        wandb.init(project='RadarNR',config=vars(args))
            
        # init model
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')
        # trainer = pl.Trainer(logger=logger, max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')

        # training
        trainer.fit(model, data_module)

        # save hist
        with open(os.path.join(args.output_path, hist_filename), 'wb') as f:
            pickle.dump(history,f)

        # save trained weights
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.output_path, model_ckpt_filename))

        # testing 
        start_t = time.perf_counter()
        trainer.test(model, data_module)
        end_t = time.perf_counter()
        print('evaluated %d point clouds in %f seconds'%(len(data_module.df_test),end_t-start_t))

        # save hist (with test output)
        with open(os.path.join(args.output_path, hist_filename), 'wb') as f:
            pickle.dump(history,f)
