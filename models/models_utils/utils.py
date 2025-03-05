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

# from sklearn.model_selection import train_test_split
from models_utils.models_visualizer import plot_confusion_mat
from models_utils.config import device, ndevice

from nuscenes import NuScenes


sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']



# nuScenes functions utils
def load_nusc(split,data_root):
    assert split in ['train','val','test','mini'], "Bad nuScenes version"

    if split=='mini':
        nusc_version = 'v1.0-mini'
    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

# DATALOADER
def get_df_split(nusc, args, data_split):
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
            if args.network_name == 'CameraNDet':
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
                    data_paths.append(os.path.join('../nuScenes',filename))
                    labels.append(0)
                    sensors_list.append(sensor)

            elif args.network_name == 'RadarNDet':
                for sensor in radar_list:
                    # Load nusc info
                    sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                    filename = sample_data['filename']
                    token = filename.split('/')[-1]

                    getOG=False

                    for noise_level in range (10,110,10):
                        synthpath = os.path.join(args.data_root,'samples',sensor,str(noise_level),token)
                        if os.path.exists(synthpath):
                            data_paths.append(synthpath)
                            labels.append(int(noise_level/10))
                            sensors_list.append(sensor)
                            getOG=True # signal flag that data is good to take from OG as well


                    if getOG:
                        data_paths.append(os.path.join('../nuScenes',filename))
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

def create_df(args):
    nusc = load_nusc('mini','../data/nuScenes')

    # init val
    n_train_scenes, n_test_scenes, n_val_scenes = args.data_split_n
    train_split = []
    test_split = []
    val_split = []

    # accumulate scene names
    for scene in nusc.scene:
        if args.network_name == 'CameraNDet':
            scene_names = [scene['name'] for scene in nusc.scene][:-3]  # accumulating scene names, removing night scenes
        elif args.network_name == 'RadarNDet':
            scene_names = [scene['name'] for scene in nusc.scene]       # accumulating scene names, using night scenes for radar

    # random scene(s) selection
    for i in range(n_train_scenes):
        idx = random.randrange(len(scene_names))
        train_split.append(scene_names.pop(idx))

    for i in range(n_test_scenes):
        idx = random.randrange(len(scene_names))
        test_split.append(scene_names.pop(idx))

    for i in range(n_val_scenes):
        idx = random.randrange(len(scene_names))
        val_split.append(scene_names.pop(idx))

    # Generating output splits
    df_train = get_df_split(nusc, args, train_split)
    df_val   = get_df_split(nusc, args, val_split)
    df_test  = get_df_split(nusc, args, test_split)

    return df_train, df_val, df_test  

def load_imgs(row):
    '''
    takes a row of a df in input
    returns a tensor with the loaded image and a tensor of its label
    '''
    data = torch.tensor(cv2.imread(row['data']), dtype=torch.float32).permute(2, 0, 1).to(device)
    labels = torch.tensor(row['labels'], dtype=torch.long).to(device)
    return data, labels

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

def batch_generator(args, df):
    indices = list(df.index)
    random.shuffle(indices)  # Shuffle data at each epoch
    
    batch = []

    if args.network_name=='CameraNDet':
        for idx in indices:
            row = df.loc[idx]

            data_tensor, label_tensor = load_imgs(row)
            
            batch.append((data_tensor, label_tensor))

            if len(batch) == args.batch_size:
                yield batch  # Yield batch when full
                batch = []  # Reset batch list
        
        if batch:  # Yield the last batch if it's not empty
            yield batch
   
    elif args.network_name=='RadarNDet':    # radar data loaded 1 by 1
        for idx in indices:
            row = df.loc[idx]

            data_tensor, label_tensor, radar_df = load_pcd(row)
            
            if len(radar_df) and not radar_df.isna().sum().any(): # length check and nan check
                yield [data_tensor, label_tensor]  # Yield batch when full


# train / test functions
def train(model,args,df_train,df_val,optimizer,loss_fct):
    # init loss history
    history = {'train_loss':[],
                'val_loss':[],
                'test_loss':[],
                'train_accuracy':[],
                'val_accuracy':[],
                'test_accuracy':[]}

    print('Training')
    for epoch in range(args.n_epochs):
        print('Epoch:',epoch,'/',args.n_epochs)
        ################Training################
        model.train()   # set model in train mode
        epoch_loss = 0
        correct_predictions=0
        total_predictions=0

        for batch_idx, batch in enumerate(batch_generator(args,df_train)):
            print(100*' ',end='\r')
            print('batch:',batch_idx, '/', round(len(df_train)/args.batch_size),end='\r')
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

        # clear up training tensor from mem
        del X_train
        torch.cuda.empty_cache()  # Clear cache after deletion

        ################VALIDATION################
        model.eval()        # eval mode
        val_loss=0
        correct_predictions_val=0
        total_predictions_val=0

        with torch.no_grad():
            for batch_idx, val_batch in enumerate(batch_generator(df_val,args.batch_size)):
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
        if args.scheduler:
            scheduler.step()

        # output values
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Save model and results
        if args.save_model:
            torch.save(model.state_dict(), args.output_path+args.sensor_type+'_model.pth')

        if args.save_hist:
            with open(args.output_path+args.sensor_type+'_model_hist.pkl', 'wb') as f:
                pickle.dump(history,f)

    print('\nFinal loss: \t train: %0.3f \t val: %0.3f'%(epoch_loss,val_loss))
    print('Final accuracy: \t train: %0.2f \t val: %0.2f'%(epoch_accuracy,val_accuracy))

    return model, history

def test(model,args,df_test,loss_fct):
    print('Testing')
    ################TESTING################
    model.eval()
    test_loss=0
    correct_predictions_test=0
    total_predictions_test=0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, test_batch in enumerate(batch_generator(df_test,args.batch_size)):
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

    # output values
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)

    plot_confusion_mat(all_preds, all_labels,args.output_path+args.sensor_type+'_model_mat.png')

    return history
