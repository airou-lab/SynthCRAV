# SynthCRAV - Realistic Noise Sythetiser for Camera-Radar Autonomous Vehicle datasets
This is the official repository for the paper *Synthetizing and Identifying Noise Levels in Autonomous Vehicle Canera-Radar Datasets* submitted to the IEEE IROS 2025 conference.<br>

**Abstract**: Detecting and tracking objects is a crucial component of any autonomous navigation method. For the past decades, object detection has yielded promising results using neural networks on various datasets. While many methods focus on performance metrics, few projects focus on improving the robustness of these detection and tracking pipelines, notably to sensor failures. In this paper we attempt to address this issue by creating a realistic synthetic data augmentation pipeline for camera-radar Autonomous Vehicle (AV) datasets. Our goal is to accurately simulate sensor failures and data deterioration due to real-world interferences. We also present our results of a baseline lightweight Noise Recognition neural network trained and tested on our augmented dataset.


## Git
Clone the repo
```shell
git clone git@github.com:airou-lab/SynthCRAV.git 

```

## Installation
Create a conda environment using python 3.8
```shell
conda init
conda create --name synthcrav python=3.8
conda activate synthcrav
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

## Data preparation
Download and extract the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).<br>
Make sure your folder architecture corresponds to:
```
SynthCRAV
├──data
|   ├──nuScenes
|   |   ├── maps
|   |   ├── samples
|   |   ├── v1.0-mini
```
We advise you to run the synthesizer on the **mini dataset** only as the generated data takes up to 40 times more space as the original. If you want to use the trainval instead, download it and name it v1.0-trainval. Then make sur to set the split argument to ```--split train```.


## Synthetic dataset
For each sensor, we generate 10 different nosie levels, going from 10% to 100%. The corrresponding folders are numbered 1 to 10. The original data is considered to be 0% noise, and thus is numbered 0. <br>
As cameras have various possible types of noises, we generate the 10 different noise levels for each noise type.<br> 
The output architecture is as follows:
```
SynthCRAV
├──data
|   ├──nuScenes
|   |   ├── maps
|   |   ├── samples
|   |   ├── v1.0-mini
|   ├──noisy_nuScenes
|   |   ├── samples
|   |   |	├── CAM_[]
|   |   |	|	├── <noise_level>
|	|   |   |	|	├── <noise_type>
|   |   |	├── RADAR_[]
|   |   |	|	├── <noise_level>
```

To run the synthetizer do:
```bash
python dataset_handler.py 
```

## Noise Recognition Model
Once the noise synthetizer has finished, you can train or test our noise recognition models.<br>
Alternatively, find pre-trained checkpoints in ckpt/.

**train/test**
```bash
python noise_classifier_cam.py
python noise_classifier_radar.py
```

