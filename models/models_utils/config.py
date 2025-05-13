import torch
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
    ndevice = torch.cuda.current_device()
    if os.getpid() == 1:
        print('\nfound CUDA device:', torch.cuda.get_device_name(ndevice))
else:
    if os.getpid() == 1:
        print('\nno CUDA installation found, using CPU')
    device = torch.device('cpu')
    ndevice = torch.cuda.current_device()
