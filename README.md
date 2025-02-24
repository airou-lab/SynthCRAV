# MatData
Versioning for custom noisy nuScenes dataset handling functions


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

