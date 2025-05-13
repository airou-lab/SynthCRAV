#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import numpy as np
import pandas as pd
import cv2
import random
import pickle
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.utils import *

from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from nuscenes.utils import splits


sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

# UTILS
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


    # Generating output splits
    df_train = get_df_split(nusc, args, sensor, list(train_split))
    df_val   = get_df_split(nusc, args, sensor, list(val_split))
    df_test  = get_df_split(nusc, args, sensor, list(test_split))

    if args.verbose:
        print('sensor:',sensor)
        print('trainval:',trainval)
        print('n_train_scenes:',n_train_scenes)
        print('n_val_scenes:',n_val_scenes)
        
        print('train_split:',list(train_split))
        print('val_split:',list(val_split))
        print('test_split:',list(test_split))

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

def load_pcd_fixed(row):
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


    # print (data.shape)
    # data = data.unsqueeze(0).transpose(1, 2)  # Shape: [1, N, C] -> [1, C, N]
    padded_data = padded_data.transpose(0, 1)  # Shape: [N, C] -> [C, N]

    # print(padded_data)
    # print(padded_data.shape)
    # input()

    return padded_data, labels, radar_df

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
        data, labels, _ =load_pcd(row)
        if not data.shape[1]:  # Skip empty point clouds
            return None
        return data, labels


class DataModule(pl.LightningDataModule):
    def __init__(self, args, sensor, batch_size=16, n_workers=4):
        super().__init__()

        nusc = load_nusc('mini','./data/default_nuScenes')  # loading nusc table
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



