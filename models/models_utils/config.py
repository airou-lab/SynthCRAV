import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    ndevice = torch.cuda.current_device()
    print('\nfound CUDA device:', torch.cuda.get_device_name(ndevice))
else:
    print('\nno CUDA installation found, using CPU')
    device = torch.device('cpu')
    ndevice = torch.cuda.current_device()
