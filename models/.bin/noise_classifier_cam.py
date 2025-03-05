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

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from ..utils.utils import *
from utils.models_visualizer import plot_confusion_mat


# from torchsummary import summary

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

if torch.cuda.is_available():
    device = torch.device('cuda')
    ndevice = torch.cuda.current_device()
    print('found CUDA device:', torch.cuda.get_device_name(ndevice))
else:
    print('no CUDA installation found, using CPU')

# DATALOADER
def get_df_split(nusc, data_split):
    '''
    For Cameras the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<noise_type>/<name.jpg>
    '''

    # output
    labels_cam = []
    data_paths_cam = []
    sensor_cam = []

    # accumulate in df:
    for scene in nusc.scene:
        if scene['name'] not in data_split:
            continue

        nusc_sample = nusc.get('sample', scene['first_sample_token'])

        while True:
            for sensor in cam_list:
                # Load nusc info
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = sample_data['filename']
                token = filename.split('/')[-1]

                getOG=False

                for noise_level in range (10,110,10):
                    for noisy_type in ['Blur', 'Gaussian_noise', 'High_exposure', 'Low_exposure']:
                        synthpath = os.path.join('./noisy_nuScenes','samples',sensor,str(noise_level),noisy_type,token)
                        if os.path.exists(synthpath):
                            labels_cam.append(int(noise_level/10))
                            data_paths_cam.append(synthpath)
                            sensor_cam.append(sensor)
                            getOG=True # signal flag that data is good to take from OG as well


                if getOG:
                    data_paths_cam.append(os.path.join('nuScenes',filename))
                    labels_cam.append(0)
                    sensor_cam.append(sensor)


            if nusc_sample['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)
    
    df = pd.DataFrame({'data':data_paths_cam,'labels':labels_cam,'sensor':sensor_cam})

    return df

def create_df(data_path):
    nusc = load_nusc('mini','./nuScenes')
    
    # init val
    train_split = []
    test_split = []
    val_split = []

    n_train_scenes = 2
    n_test_scenes = 1
    n_val_scenes = 1

    # accumulate scene names
    for scene in nusc.scene:
        scene_names = [scene['name'] for scene in nusc.scene][:-3]  # accumulating scene names, removing night scenes

    for i in range(n_train_scenes):
        idx = random.randrange(len(scene_names))
        train_split.append(scene_names.pop(idx))

    for i in range(n_test_scenes):
        idx = random.randrange(len(scene_names))
        test_split.append(scene_names.pop(idx))

    for i in range(n_val_scenes):
        idx = random.randrange(len(scene_names))
        val_split.append(scene_names.pop(idx))


    df_train = get_df_split(nusc, train_split)
    df_val   = get_df_split(nusc, val_split)
    df_test  = get_df_split(nusc, test_split)

    print('df_train:',df_train)
    print('df_val:',df_val)
    print('df_test:',df_test)

    return df_train, df_val, df_test  

def load_imgs(row):
    '''
    takes a row of a df in input
    returns a tensor with the loaded image and a tensor of its label
    '''
    data = torch.tensor(cv2.imread(row['data']), dtype=torch.float32).permute(2, 0, 1).to(device)
    labels = torch.tensor(row['labels'], dtype=torch.long).to(device)
    return data, labels

def batch_generator(df, batch_size):
    indices = list(df.index)
    random.shuffle(indices)  # Shuffle data at each epoch
    
    batch = []
    for idx in indices:
        row = df.loc[idx]

        data_tensor, label_tensor = load_imgs(row)
        
        batch.append((data_tensor, label_tensor))

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

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 


        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #


        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #


        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(conv_k,conv_k), \
        #                        stride=1, padding=0, bias=True, padding_mode = 'zeros'),
        #                        nn.ReLU(),
        #                        nn.Dropout2d(p=dropout_prob),
        #                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # image_size=np.floor((image_size-2)/2) # 65x118



        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(int(8*image_size[0]*image_size[1]),output_size))

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.head(x)
        return x


# train / test functions
def train_model (model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler):
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
            for batch_idx, val_batch in enumerate(batch_generator(df_val,batch_size)):
                # Load data
                X_val = torch.stack([x[0] for x in val_batch]).detach()
                labels_val = torch.stack([x[1] for x in val_batch]).detach()

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

        # Save model and results
        torch.save(model.state_dict(), './ckpt/cam_model.pth')
        with open("cam_model_hist.pkl", "wb") as f:
            pickle.dump(history,f)

    print('\nFinal loss: \t train: %0.3f \t val: %0.3f'%(epoch_loss,val_loss))
    print('Final accuracy: \t train: %0.2f \t val: %0.2f'%(epoch_accuracy,val_accuracy))

    return model, history

def test_model(model,df_test,loss_fct,history,batch_size=8):
    print('Testing')
    ################TESTING################
    model.eval()
    test_loss=0
    correct_predictions_test=0
    total_predictions_test=0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, test_batch in enumerate(batch_generator(df_test,batch_size)):
            # Load data
            X_test = torch.stack([x[0] for x in test_batch])
            labels_test = torch.stack([x[1] for x in test_batch])

            test_pred = model(X_test)
            test_loss += loss_fct(test_pred,labels_test).item()

             # Calculate accuracy
            _, predicted = torch.max(test_pred, 1)
            correct_predictions_test += (predicted == labels_test).sum().item()
            total_predictions_test += labels_test.size(0)

            all_preds.extend(predicted.cpu().numpy())  # Convert to numpy and store
            all_labels.extend(labels_test.cpu().numpy())  # Convert to numpy and store
        
    test_accuracy = 100 * (correct_predictions_test/total_predictions_test)

    print('Test loss: %0.3f \t | \taccuracy:%0.2f'%(test_loss,test_accuracy))

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    plot_confusion_mat(all_preds, all_labels,'cam_model_mat.png')


    # output values
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)

    return history



if __name__ == '__main__':
    # df = create_df('./noisy_nuScenes/samples')
    # df_train, df_test, df_val = get_df_split(df)

    df_train, df_val, df_test = create_df('./noisy_nuScenes/samples')

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
    n_epochs = 20
    lr = 1e-4
    batch_size = 16
    
    # init
    model = simple_CNN(image_shape=img_shape, output_size=n_labels, conv_k=3, dropout_prob=0).to(device)
    print('model:\n',model)

    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-20)
    scheduler=None

    #train
    model, history = train_model(model,n_epochs,batch_size,df_train,df_val,optimizer,loss_fct,scheduler)

    #test
    history = test_model(model,df_test,loss_fct,history,batch_size)

    with open("cam_model_hist.pkl", "wb") as f:
        pickle.dump(history,f)
