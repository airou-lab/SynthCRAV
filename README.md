# SynthCRAV - Realistic Noise Sythetiser for Camera-Radar Autonomous Vehicle datasets
This is the official repository for the paper *Synthetizing and Identifying Noise Levels in Autonomous Vehicle Canera-Radar Datasets* 



## Git
Clone the repo
```shell
git clone git@github.com:airou-lab/MatData.git 

```

## Installation
Create a conda environment using python 3.8
```shell
conda init
conda create --name dataset python=3.8
conda activate dataset
```
Then install the packages
```shell
pip install nuscenes-devkit -U
pip install open3d
pip install pandas
```

Note: <br> 
The nuscenes-devkit install will also take care of installing the correct versions for numpy, pyquaternion, opencv and matplotlib so it is advised to install it first.

install pytorch:
```shell
# On windows with cuda 12.8 (tested with Geforce GTX 1050 mobile):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# On linux with cuda 12.3 (tested with Geforce RTX 3060):
pip3 install torch torchvision torchaudio
```

