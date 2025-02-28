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

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
# from torchsummary import summary

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

if torch.cuda.is_available:
    device = torch.device('cuda')
    ndevice = torch.cuda.current_device()
    print('found CUDA device:', torch.cuda.get_device_name(ndevice))
else:
    print('no CUDA installation found, using CPU')


# DATALOADER

def create_df(data_path):
    '''
    For Radars the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<name.pcd>
    For Cameras the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<noise_type>/<name.jpg>
    '''
    labels_cam = []
    data_paths_cam = []
    sensor_cam = []

    labels_radar = []
    data_paths_radar = []
    sensor_radar = []
    
    for sensor in os.listdir(data_path):
        for noise_level in os.listdir(os.path.join(data_path,sensor)):
            if 'CAM' in sensor: 
                for noise_type in os.listdir(os.path.join(data_path,sensor,noise_level)):
                    for item in os.listdir(os.path.join(data_path,sensor,noise_level,noise_type)):
                        data_paths_cam.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                        labels_cam.append(noise_level)
                        sensor_cam.append(sensor)
            else:
                for item in os.listdir(os.path.join(data_path,sensor,noise_level)):
                    data_paths_radar.append(os.path.join(data_path,sensor,noise_level,item))
                    labels_radar.append(noise_level)
                    sensor_radar.append(sensor)

    # original dataset --> no added noise
    for sensor in os.listdir(os.path.join('nuScenes','samples')):
        for item in os.listdir(os.path.join('nuScenes','samples',sensor)):
            if 'CAM' in sensor:
                data_paths_cam.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                labels_cam.append(0.0)
                sensor_cam.append(sensor)
            elif 'RADAR' in sensor:
                if os.path.exists(os.path.join(data_path,'samples',sensor,'10',item)):
                    # Making sure we are gathering only non-empty data points. 
                    # The dataset handler skipped those so they arent in noisy folders.
                    data_paths_radar.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                    labels_radar.append(0.0)
                    sensor_radar.append(sensor)

    df_cam = pd.DataFrame({'data':data_paths_cam,'labels':labels_cam,'sensor':sensor_cam})
    df_radar = pd.DataFrame({'data':data_paths_radar,'labels':labels_radar,'sensor':sensor_radar})

    return df_cam, df_radar

def get_df_split(df,sensortype):
    # split is 80/10/10
    r_train = 0.8
    r_test = 0.1
    r_val = 0.1

    df_train = pd.DataFrame(columns = df.columns)
    df_test = pd.DataFrame(columns = df.columns)
    df_val = pd.DataFrame(columns = df.columns)
    
    if sensortype == 'cam':
        for sensor in cam_list:
            subdf = df.loc[df['sensor']==sensor]

            # separating training set (80%) from dataset (20%) 
            df_temp, df_train_temp = train_test_split(subdf, test_size=r_train, random_state=None)
            df_train = pd.concat([df_train,df_train_temp], ignore_index=True)

            # further separating leftovers in 2 (10% each) splits: test and val
            df_test_temp, df_val_temp =  train_test_split(df_temp, test_size=0.5, random_state=None)
            df_test =  pd.concat([df_test,df_test_temp], ignore_index=True)
            df_val = pd.concat([df_val,df_val_temp], ignore_index=True)
    
    elif sensortype == 'radar':
        for sensor in radar_list:
            subdf = df.loc[df['sensor']==sensor]

            df_train_temp, df_temp = train_test_split(subdf, test_size=r_train, random_state=None)
            df_train = pd.concat([df_train,df_train_temp], ignore_index=True)

            df_test_temp, df_val_temp =  train_test_split(df_temp, test_size=0.5, random_state=None)
            df_test =  pd.concat([df_test,df_test_temp], ignore_index=True)
            df_val = pd.concat([df_val,df_val_temp], ignore_index=True)


    return df_train, df_test, df_val

def load_imgs(batch):
    '''
    takes a df in input (batch)
    returns a tensor with the loaded images
    '''
    data_list = []
    label_list = batch['labels'].to_numpy()

    # extract images
    for i in range(len(batch)):
        filepath = batch.loc[i,'data']

        data_list.append(cv2.imread(filepath))
    

    # converting to torch tensor
    
    labels = torch.tensor(label_list, dtype=torch.float32).permute(2, 0, 1).to(device)
    data = torch.tensor(np.array(data_list), dtype=torch.long).to(device)
    return data, labels

def load_pcd():
    pass

def batch_generator(df, batch_size):
    indices = list(df.index)
    random.shuffle(indices)  # Shuffle data at each epoch
    
    batch = []
    for idx in indices:
        row = df.loc[idx]
        img_tensor, label_tensor = load_imgs(row)

        batch.append((img_tensor, label_tensor))

        if len(batch) == batch_size:
            yield batch  # Yield batch when full
            batch = []  # Reset batch list
    
    if batch:  # Yield the last batch if it's not empty
        yield batch

# MODELS
class simple_CNN(torch.nn.Module):
    def __init__(self,image_size,n_conv_layers,output_size):
        super(simple_CNN, self).__init__()
        self.image_size = image_size
        self.n_conv_layers = n_conv_layers
        self.output_size = output_size

        # self.input = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU())
        # self.hidden = nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU())
        
        self.head = nn.Sequential(nn.Linear(128,output_size))

    def forward(self, x):
        # input layer connect to data
        x = nn.Conv2d(self.image_size,)

        for i in range(self.n_conv_layers):
            pass

        # x = self.input(x)
        # for i in range(5):
        #     # hidden layers
        #     x = self.hidden(x)



        # output layer / head
        x = self.head(x)

        return x

# train / test functions
def train_model (model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler):
    # init loss history
    history = dict()

    print('Training')
    for epoch in range(n_epochs):
        ################Training################
        model.train()   # set model in train mode

        for batch_idx, (img_batch, labels) in enumerate(df_train):
            pass

        # Load data
        X_train, y_train = load_imgs(df_train)
        X_val, y_val = load_imgs(df_val)

        # generate predictions
        pred = model(X_train)

        # calculate loss
        loss = loss_fct(pred,y_train)

        # backprop loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ################VALIDATION################
        # val_loss = test_model(model,val_split['X'],val_split['y'],loss_fct)  # using test function on validation dataset
        val_loss=0

        print('\rEpoch: %d \t train loss: %0.3f \t val loss: %0.3f'%(epoch+1, loss.item(),val_loss.item()), end='',flush=True)
        
        # scheduler step
        if scheduler:
            scheduler.step()


        # output values
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())

    print('\nFinal loss: \t train: %0.3f \t val: %0.3f'%(loss,val_loss))

    return model, history

def test_model(model,X_test,y_test,loss_fct):
    model.eval()    # set model in evaluation mode (no training)
    # Load data
    X = torch.Tensor(X_test).to(device)
    y = torch.Tensor(y_test).to(device)
    # generate predictions
    pred = model(X)
    # loss
    loss = loss_fct(pred,y)

    return loss   




if __name__ == '__main__':
    df_cam, df_radar = create_df('./noisy_nuscenes/samples')
    
    # Cameras
    df_train, df_test, df_val = get_df_split(df_cam,'cam')
    n_train = len(df_train)
    n_test = len(df_test)
    n_val = len(df_val)


    # param
    n_epochs=10
    lr = 1e-5
    batch_size = 32
    
    # init
    model = simple_CNN().to(device)
    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-20)
    scheduler=None

    # train
    train_model(model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler)
    