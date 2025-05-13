#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import pickle
import argparse
import wandb
from copy import copy

import torch
import torch.multiprocessing as mp
from torch import nn
import torchviz
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


# from torchsummary import summary

from models.models_utils.utils import *
from models.models_utils.config import device, ndevice
from models.models import RadarNDet, CameraNDet

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']


def get_labels(data_module):
    # Loading labels
    labels = copy(data_module.df_train['labels']).drop_duplicates().sort_values().reset_index(drop=True)
    n_labels = len(labels)

    return labels, len(labels)


def create_parser():

    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument('--nusc_root', type=str, default='./data/default_nuScenes/', help='Original nuScenes data folder')
    parser.add_argument('--split', type=str, default='mini', help='train/val/test/mini')
    parser.add_argument('--data_root', type=str, default='./data/noisy_nuScenes/', help='Synth data folder')
    parser.add_argument('--output_path', type=str, default='./ckpt/', help='checkpoint save path')

    # Network selection
    parser.add_argument('--sensor', type=str, default='CAM', help='CAM | RADAR')
    parser.add_argument('--network_name', type=str, default='', help='Internal name of network. Do not atrribute a value.')
    parser.add_argument('--ntrain', type=float, default=0.7, help='Set train set size')

    # misc
    # parser.add_argument('--scheduler', type=str, default=None, help='Scheduler setup. Unsupported for now')  # TODOs
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
    parser.add_argument('--dropout2d', type=float, default=0.1, help='2D convolution dropout rate (cam)')
    parser.add_argument('--dropout1d', type=float, default=0.1, help='1D convolution dropout rate (radar)')

    # Verbosity level
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')

    return parser

def check_args(args):
    assert args.sensor_type in ['camera', 'radar'], 'wrong sensor type'
    
    assert len(args.data_split_n) == 3, 'Error, data_split_n is a list of 3 elements, got %d'%(len(args.data_split_n))
    assert len(args.img_shape) == 3, 'Error, img_shape is a list of 3 elements (WxHxD), got %d'%(len(args.img_shape))

    assert args.data_split_n[0] > 0,'train split cannot be empty'
    assert args.data_split_n[1] > 0,'val split cannot be empty'
    assert args.data_split_n[2] > 0,'test split cannot be empty'

    assert args.data_split_n[0] + args.data_split_n[1] <= 8,'train/val splits too large, max total length is 8'
    assert args.data_split_n[2] <= 2,'max test spllit size is 2'

    if args.sensor_type=='camera':
        assert args.data_split_n[0] + args.data_split_n[1] <=5,'train/val splits too large, max total length is 5 for camera noise recognition (night scenes removed)'



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
    # check_args(args)

    if args.n_workers>0:
        mp.set_start_method('spawn', force=True)

    history = history = {'train_loss':[],
                            'val_loss':[],
                            'test_loss':[],
                            'train_accuracy':[],
                            'val_accuracy':[],
                            'test_accuracy':[],
                            'test_results':[]}


    if args.sensor =='CAM':
        # Load data
        data_module = DataModule(args=args, sensor='CAM',batch_size=args.batch_size,n_workers=args.n_workers)    
        # Loading labels
        labels, n_labels = get_labels(data_module)
        # init model
        model = CameraNDet(image_shape=[900,1600,3], output_size=n_labels, history=history, conv_k=args.conv_k, dropout_prob=args.dropout2d, lr=args.lr).to(device)
        model_ckpt_filename = 'camera_model.pth'
        hist_filename = 'camera_model_hist.pkl'
        # logger = TensorBoardLogger('lightning_logs', name='camera_noise_classifier')
        logger = WandbLogger(project="CameraNR", name='EXP0')

    elif args.sensor=='RADAR':
        # Load data
        data_module = DataModule(args=args, sensor='RADAR',batch_size=1,n_workers=args.n_workers)
        # Loading labels
        labels, n_labels = get_labels(data_module)
        # init model
        model = RadarNDet(n_channels=18, output_size=n_labels, history=history, dropout_prob=args.dropout1d, lr=args.lr).to(device)
        model_ckpt_filename = 'radar_model.pth'
        hist_filename = 'radar_model_hist.pkl'
        # logger = TensorBoardLogger('lightning_logs', name='radar_noise_classifier')
        # logger = WandbLogger(project="RadarNR", name='EXP0')




    if args.load_checkpoint:
        model.load_state_dict(torch.load(os.path.join(args.output_path, model_ckpt_filename)))
        with open(os.path.join(args.output_path,hist_filename), 'rb') as f:
            model.history = pickle.load(f)

        
    # init model
    # trainer = pl.Trainer(logger=logger, max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')
    trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu')
    
    if not args.eval:
        # training
        wandb.init(project=args.sensor+'NR',config=vars(args))
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


'''
default arguments:
CameraNDet, conv_k=3, dropout=0, lr=1e-4, nepochs=100, batch_size=16, no train, no test


Run with :

python noise_classifier.py --sensor_type camera --data_split_n 2 1 1 --lr 1e-3 --n_epochs 1 --batch_size 16 \
                        --conv_k 3 --dropout2d 0 \
                        --train --save_model --save_hist -v

python noise_classifier.py --sensor_type radar --data_split_n 2 1 1 --lr 1e-4 --n_epochs 1 --dropout1d 0.1\
                                             --train --save_model --save_hist -v
'''