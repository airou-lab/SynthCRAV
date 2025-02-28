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
from utils.utils import *
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
    
    print('Creating dataset')

    for sensor in os.listdir(data_path):
        for noise_level in os.listdir(os.path.join(data_path,sensor)):
            if 'CAM' in sensor: 
                for noise_type in os.listdir(os.path.join(data_path,sensor,noise_level)):
                    for item in os.listdir(os.path.join(data_path,sensor,noise_level,noise_type)):
                        data_paths_cam.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                        labels_cam.append(int(noise_level[:-1])) # encoding noise value to an integer (1,2,3,4,5,6,7,8,9,10])
                        sensor_cam.append(sensor)
            else:
                for item in os.listdir(os.path.join(data_path,sensor,noise_level)):
                    data_paths_radar.append(os.path.join(data_path,sensor,noise_level,item))
                    labels_radar.append(int(noise_level[:-1])) # encoding noise value to an integer (1,2,3,4,5,6,7,8,9,10])
                    sensor_radar.append(sensor)

    # original dataset --> no added noise
    for sensor in os.listdir(os.path.join('nuScenes','samples')):
        for item in os.listdir(os.path.join('nuScenes','samples',sensor)):
            if 'CAM' in sensor:
                data_paths_cam.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                labels_cam.append(0)
                sensor_cam.append(sensor)
            elif 'RADAR' in sensor:
                if os.path.exists(os.path.join(data_path,'samples',sensor,'10',item)):
                    # Making sure we are gathering only non-empty data points. 
                    # The dataset handler skipped those so they arent in noisy folders.
                    data_paths_radar.append(os.path.join(data_path,sensor,noise_level,noise_type,item))
                    labels_radar.append(0)
                    sensor_radar.append(sensor)

    df_cam = pd.DataFrame({'data':data_paths_cam,'labels':labels_cam,'sensor':sensor_cam})
    df_radar = pd.DataFrame({'data':data_paths_radar,'labels':labels_radar,'sensor':sensor_radar})

    return df_cam, df_radar

def get_df_split(df,sensortype):
    # split is 80/10/10
    r_train = 0.8
    r_test = 0.1
    r_val = 0.1

    print('Creating splits')

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

    print('df_train:',df_train)
    print('df_test:',df_test)
    print('df_val:',df_val)

    return df_train, df_test, df_val

def load_imgs(row):
    '''
    takes a row of a df in input
    returns a tensor with the loaded image and a tensor of its label
    '''
    data = torch.tensor(cv2.imread(row['data']), dtype=torch.float32).permute(2, 0, 1).to(device)
    labels = torch.tensor(row['labels'], dtype=torch.long).to(device)
    return data, labels

# TODO: nan data case check
def load_pcd(row):
    '''
    takes a df in input (batch)
    returns a tensor with the loaded images
    '''
    labels = torch.tensor(row['labels'], dtype=torch.long).to(device)
    radar_df, types_str = decode_pcd_file(row['data'],verbose=False)

    data = convert_radardf_to_tensor(radar_df,types_str).to(device)

    return data, labels

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
    def __init__(self,image_shape,output_size,conv_k,dropout_prob=0):
        super(simple_CNN, self).__init__()
        
        image_size = image_shape[:2] # 1080 x 1920
        in_channels = image_shape[2] # 3

        self.input = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) # 539x959

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) # 268x478
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) # 133x238
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) # 65x118



        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(int(128*image_size[0]*image_size[1]),output_size))

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.head(x)
        return x

# train / test functions
def train_model (model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler):
    # init loss history
    history = dict()

    print('Training')
    for epoch in range(n_epochs):
        print('Epoch:',epoch)
        ################Training################
        model.train()   # set model in train mode
        epoch_loss = 0

        for batch_idx, batch in enumerate(batch_generator(df_train,batch_size)):
            print(100*' ',end='\r')
            print('batch:',batch_idx, '/', round(len(df_train)/batch_size),end='\r')
            # Load data
            X_train = torch.stack([x[0] for x in batch])
            labels_train = torch.stack([x[1] for x in batch])
            
            # generate predictions (i.e noise values)
            pred = model(X_train)

            # calculate loss
            loss = loss_fct(pred,labels_train)

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        


        ################VALIDATION################
        model.eval()

        val_data = batch_generator(df_val,len(df_val)) # using batch generator to generate the full val dataset
        X_val = torch.stack([x[0] for x in val_data])
        labels_val = torch.stack([x[1] for x in val_data])

        val_pred = model(X_val)
        val_loss = loss_fct(val_pred,labels_val)



        # scheduler step
        if scheduler:
            scheduler.step()

        print('\ntrain loss: %0.3f \t val loss: %0.3f'%(epoch+1, loss.item(),val_loss.item()))

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

    img_shape = np.array(cv2.imread(df_train.loc[0,'data']).shape)
    print('image shape:',img_shape)

    labels = df_train['labels'].drop_duplicates().sort_values().reset_index(drop=True)
    n_labels = len(labels)
    print('labels:',labels)
    print('n_labels:',n_labels)

    # param
    n_epochs=10
    lr = 1e-5
    batch_size = 32
    
    # init
    model = simple_CNN(image_shape=img_shape, output_size=n_labels,conv_k=3,dropout_prob=0).to(device)
    print('model:\n',model)

    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-20)
    scheduler=None

    # train cam
    train_model(model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler)
    
