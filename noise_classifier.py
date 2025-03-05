#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import pickle
import argparse
from copy import copy

import torch
from torch import nn
# from torchsummary import summary

from models.models_utils.utils import *
from models.models_utils.config import device, ndevice
from models.models import RadarNDet, CameraNDet

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']



def create_parser():

    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument('--nusc_root', type=str, default='./data/nuScenes/', help='Original nuScenes data folder')
    parser.add_argument('--split', type=str, default='mini', help='train/val/test/mini')
    parser.add_argument('--data_root', type=str, default='./data/noisy_nuScenes/', help='Synth data folder')
    parser.add_argument('--output_path', type=str, default='./ckpt/', help='Synth data folder')

    # Network selection
    parser.add_argument('--sensor_type', type=str, default='camera', help='Allows to train separately or together (camera, radar, both)')
    parser.add_argument('--network_name', type=str, default='', help='Internal name of network. Do not atrribute a value.')
    parser.add_argument('--data_split_n', type=int, nargs='+', default=[2,1,1], help='sets amount of scene per splits (train/val/test)')

    # misc
    parser.add_argument('--img_shape', type=int, nargs='+', default=[900,1600,3], help='nuScenes image size')
    parser.add_argument('--n_cols', type=int, default=18, help='Number of columns in a radar point cloud')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler setup. Unsupported for now')  # TODOs

    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default= 100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # actions
    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--test', action='store_true', default=False, help='test model')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='load checkpoint at <output_path>/<sensor_type>+_model.pth')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model at <output_path>/<sensor_type>+_model.pth')
    parser.add_argument('--save_hist', action='store_true', default=False, help='save model history at <output_path>/<sensor_type>+_hist.pkl')


    # network parameters
    parser.add_argument('--conv_k', type=int, default=3, help='2D convolution kernel')
    parser.add_argument('--dropout2d', type=float, default=0, help='2D convolution dropout rate (cam)')
    parser.add_argument('--dropout1d', type=float, default=0, help='1D convolution dropout rate (radar)')

    # Verbosity level
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')

    return parser

def check_args(args):
    assert args.sensor_type in ['camera', 'radar'], 'wrong sensor type'
    
    assert len(args.data_split_n) == 3, 'Error, data_split_n is a list of 3 elements, got %d'%(len(args.data_split_n))
    assert len(args.img_shape) == 3, 'Error, img_shape is a list of 3 elements (WxHxD), got %d'%(len(args.img_shape))

    assert args.network_name == '','--network_name should be an empty string'

    assert args.conv_k >= 3,'min conv size is 3'
    assert args.dropout2d >= 0 and args.dropout2d <= 1 ,'dropout2d is between 0 and 1'
    assert args.dropout1d >= 0 and args.dropout1d <= 1 ,'dropout1d is between 0 and 1'



    # init
    if args.sensor_type=='camera':
        args.network_name='CameraNDet'
    elif args.sensor_type=='radar':
        args.network_name='RadarNDet'
        args.batch_size = 1
    else:
        exit('unknown sensor')


    print(args)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Load data
    df_train, df_val, df_test = create_df(args)

    n_train = len(df_train)
    n_test = len(df_test)
    n_val = len(df_val)

    # Loading labels
    labels = copy(df_train['labels']).drop_duplicates().sort_values().reset_index(drop=True)
    n_labels = len(labels)
    
    # init
    if args.sensor_type=='camera':
        model = CameraNDet(image_shape=args.img_shape, output_size=n_labels, conv_k=args.conv_k, dropout_prob=args.dropout2d).to(device)

    elif args.sensor_type=='radar':
        model = RadarNDet(args.n_cols,n_labels,dropout_prob=args.dropout1d).to(device)

    if args.verbose:
        print('df_train:\n',df_train)
        print('df_val:\n',df_val)
        print('df_test:\n',df_test)
        
        print('labels:\n',labels)
        print('n_labels:',n_labels)

        print('image shape:',args.img_shape)
        print('point cloud features:',args.n_cols)

        print('model:\n',model)

    
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.output_path+args.sensor_type+'_model.pth'))


    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-20)

    #train
    if args.train:
        model, history = train(model,args,df_train,df_val,optimizer,loss_fct)

    #test
    if args.test:
        history = test(model,args,df_test,loss_fct,history)


    if args.save_hist:
        with open(args.output_path+args.sensor_type+'_model_hist.pkl', 'wb') as f:
            pickle.dump(history,f)



'''
default arguments:
CameraNDet, conv_k=3, dropout=0, lr=1e-4, nepochs=100, batch_size=16


Run with :

python noise_classifier.py --sensor_type camera --data_split_n 2 1 1 --lr 1e-4 --n_epochs 1 --batch_size 16 \
                        --conv_k 3 --dropout2d 0 \
                        --train --save_model --save_hist

python noise_classifier.py --sensor_type radar --data_split_n 2 1 1 --lr 1e-4 --n_epochs 1 --dropout1d 0\
                                             --train --save_model --save_hist




'''