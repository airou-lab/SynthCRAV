#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------
import numpy as np
import torch
from torch import nn

from models.models_utils.config import device, ndevice

## MODEL INSTANCES

# Radar noise classifier
class RadarNDet(torch.nn.Module):
    def __init__(self,n_channels, n_labels,dropout_prob=0):
        super(RadarNDet, self).__init__()

        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        # self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
        #                             nn.ReLU(),
        #                             nn.Dropout1d(p=dropout_prob))

        
        self.fc1 = nn.Sequential(nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.fc2 = nn.Sequential(nn.Linear(64,32),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.head = nn.Sequential(nn.Linear(32,n_labels))
        # no softmax because we use cross entropy loss


    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        x = torch.mean(x, dim=2)  # global average pooling

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.head(x)

        return x


# Camera noise classifier
# TODO : try using depthwise convolution layers for lighter network
class CameraNDet(torch.nn.Module):
    def __init__(self,image_shape,output_size,conv_k,dropout_prob=0):
        super(CameraNDet, self).__init__()
        
        image_size = np.array(image_shape[:2]) # 1080 x 1920
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
