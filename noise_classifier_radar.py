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
    '''
    labels_radar = []
    data_paths_radar = []
    sensor_radar = []
    
    print('Creating dataset')

    for sensor in os.listdir(data_path):
        for noise_level in os.listdir(os.path.join(data_path,sensor)):
            if 'RADAR' in sensor:                
                for item in os.listdir(os.path.join(data_path,sensor,noise_level)):
                    data_paths_radar.append(os.path.join(data_path,sensor,noise_level,item))
                    labels_radar.append(int(noise_level[:-1])) # encoding noise value to an integer (1,2,3,4,5,6,7,8,9,10])
                    sensor_radar.append(sensor)

    # original dataset --> no added noise
    for sensor in os.listdir(os.path.join('nuScenes','samples')):
        for item in os.listdir(os.path.join('nuScenes','samples',sensor)):
            if 'RADAR' in sensor:
                if os.path.exists(os.path.join(data_path,sensor,'10',item)):
                    # Making sure we are gathering only non-empty data points. 
                    # The dataset handler skipped those so they arent in noisy folders.
                    data_paths_radar.append(os.path.join('nuScenes','samples',sensor,item))
                    labels_radar.append(0)
                    sensor_radar.append(sensor)

    df_radar = pd.DataFrame({'data':data_paths_radar,'labels':labels_radar,'sensor':sensor_radar})

    return df_radar

def get_df_split(df):
    # split is 80/10/10
    r_train = 0.8
    r_test = 0.1
    r_val = 0.1

    print('Creating splits')

    df_train = pd.DataFrame(columns = df.columns)
    df_test = pd.DataFrame(columns = df.columns)
    df_val = pd.DataFrame(columns = df.columns)
        
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


# TODO: nan data case check
def load_pcd(row):
    '''
    takes a df in input (batch)
    returns a tensor with the loaded images
    '''
    labels = torch.tensor([row['labels']], dtype=torch.long).to(device)
    radar_df, types_str = decode_pcd_file(row['data'],verbose=False)

    data = convert_radardf_to_tensor(radar_df,types_str).to(device)
    data = data.unsqueeze(0).transpose(1, 2)  # Shape: [1, N, C] -> [1, C, N]

    return data, labels, radar_df

def batch_generator(df):
    indices = list(df.index)
    random.shuffle(indices)  # Shuffle data at each epoch
    
    batch = []
    for idx in indices:
        row = df.loc[idx]

        data_tensor, label_tensor, radar_df = load_pcd(row)
        
        if len(radar_df):
            yield [data_tensor, label_tensor, radar_df]  # Yield batch when full

# MODELS
class simple_RadarNDet(torch.nn.Module):
    def __init__(self,n_channels, n_labels,dropout_prob=0):
        super(simple_RadarNDet, self).__init__()

        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        
        self.fc1 = nn.Sequential(nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.fc2 = nn.Sequential(nn.Linear(64,16),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.head = nn.Sequential(nn.Linear(16,n_labels))
        # no softmax because we use cross entropy loss


    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)

        x = torch.mean(x, dim=2)  # global average pooling

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.head(x)

        return x


# train / test functions
def train_model (model,n_epochs,df_train,df_val,optimizer,loss_fct,scheduler):
    # init loss history
    history = {'train_loss':[],
                'val_loss':[],
                'test_loss':[],
                'train_accuracy':[],
                'val_accuracy':[],
                'test_accuracy':[]}

    print('Training')
    for epoch in range(n_epochs):
        print('Epoch:',epoch)
        ################Training################
        model.train()   # set model in train mode
        epoch_loss = 0
        correct_predictions=0
        total_predictions=0

        for batch_idx, batch in enumerate(batch_generator(df_train)):
            print(100*' ',end='\r')
            print('batch:',batch_idx, '/', round(len(df_train)),end='\r')
            
            # Load data
            X_train, labels_train, radar_df = batch             #radar_df is for debug / display
            
            # generate predictions (i.e noise values)
            pred = model(X_train)            

            # calculate loss
            loss = loss_fct(pred,labels_train)

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(pred, 1)
            correct_predictions += (predicted == labels_train).sum().item()
            total_predictions += labels_train.size(0)

            # print('running loss: %0.3f'%(loss.item()),end='\r')

        epoch_accuracy = 100 * (correct_predictions/total_predictions)
        
        print('\nEpoch train loss: %0.3f \t | \taccuracy:%0.2f'%(epoch_loss, epoch_accuracy))

        del X_train
        torch.cuda.empty_cache()  # Clear cache after deletion

        ################VALIDATION################
        model.eval()
        val_loss=0
        correct_predictions_val=0
        total_predictions_val=0

        with torch.no_grad():
            for batch_idx, val_batch in enumerate(batch_generator(df_val)):
                print(100*' ',end='\r')
                print('val_batch:',batch_idx, '/', round(len(df_val)),end='\r')
                # Load data
                X_val, labels_val, _ = val_batch

                val_pred = model(X_val)
                val_loss += loss_fct(val_pred,labels_val).item()

                # Calculate accuracy
                _, predicted = torch.max(val_pred, 1)
                correct_predictions_val += (predicted == labels_val).sum().item()
                total_predictions_val += labels_val.size(0)

            val_accuracy = 100 * (correct_predictions_val/total_predictions_val)

        print('Epoch val loss: %0.3f \t\t | \taccuracy:%0.2f'%(val_loss, val_accuracy))


        # scheduler step
        if scheduler:
            scheduler.step()

        # output values
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

    print('\nFinal loss: \t train: %0.3f \t val: %0.3f'%(epoch_loss,val_loss))
    print('Final accuracy: \t train: %0.2f \t val: %0.2f'%(epoch_accuracy,val_accuracy))

    return model, history

def test_model(model,df_test,loss_fct,history):
    print('Testing')
    ################TESTING################
    model.eval()
    test_loss=0
    correct_predictions_test=0
    total_predictions_test=0

    with torch.no_grad():
        for batch_idx, test_batch in enumerate(batch_generator(df_test)):
            print(100*' ',end='\r')
            print('test_batch:',batch_idx, '/', round(len(df_val)),end='\r')
            # Load data
            X_test, labels_test,_ = test_batch

            test_pred = model(X_test)
            test_loss += loss_fct(test_pred,labels_test).item()

             # Calculate accuracy
            _, predicted = torch.max(test_pred, 1)
            correct_predictions_test += (predicted == labels_test).sum().item()
            total_predictions_test += labels_test.size(0)
        
        test_accuracy = 100 * (correct_predictions_test/total_predictions_test)

    print('Test loss: %0.3f \t | \taccuracy:%0.2f'%(test_loss,test_accuracy))


    # output values
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)

    return history



if __name__ == '__main__':
    df = create_df('./noisy_nuScenes/samples')
    
    df_train, df_test, df_val = get_df_split(df)

    # df_train = df_train.head(8000)
    # df_test = df_train.head(1000)
    # df_val = df_train.head(1000)

    # n_train = len(df_train)
    # n_test = len(df_test)
    # n_val = len(df_val)
    
    radar_df_0, _ = decode_pcd_file(df_train.loc[0,'data'],verbose=False)
    n_cols = len(radar_df_0.loc[0])

    print('number of features:',n_cols)

    labels = df_train['labels'].drop_duplicates().sort_values().reset_index(drop=True)
    n_labels = len(labels)
    print('labels:',labels)
    print('n_labels:',n_labels)

    # param
    n_epochs=10
    lr = 1e-5
    
    # init
    model = simple_RadarNDet(n_cols,n_labels,dropout_prob=0).to(device)
    print('model:\n',model)

    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-20)
    scheduler=None

    #train
    model, history = train_model(model,n_epochs,df_train,df_val,optimizer,loss_fct,scheduler)    
    torch.save(model.state_dict(), './ckpt/radar_model.pth')

    #test
    test_model(model,df_test,loss_fct,history)

    with open("radar_model_hist.pkl", "wb") as f:
        pickle.dump(history,f)

