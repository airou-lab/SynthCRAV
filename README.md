# SynthCRAV - Realistic Noise Sythetiser for Camera-Radar Autonomous Vehicle datasets
This is the official repository for the paper *Synthetizing and Identifying Noise Levels in Autonomous Vehicle Canera-Radar Datasets* submitted to the IEEE IROS 2025 conference.<br>

**Abstract**: Detecting and tracking objects is a crucial component of any autonomous navigation method. For the past decades, object detection has yielded promising results using neural networks on various datasets. While many methods focus on performance metrics, few projects focus on improving the robustness of these detection and tracking pipelines, notably to sensor failures. In this project, we attempt to address this issue by creating a realistic synthetic data augmentation pipeline for camera-radar Autonomous Vehicle (AV) datasets. Our goal is to accurately simulate sensor failures and data deterioration due to real-world interferences. We also present our results of a baseline lightweight Noise Recognition neural network trained and tested on our augmented dataset.<br>
We present the first camera data augmentation method to simulate image blurring, overexposure, underexposure, and normally distributed additive noise. Our radar noise synthetizer is the first physics-based method to simulate what a radar point cloud would have looked like if there was more noise at the time of the measure. Since we work on the images and point cloud directly, our method does not require additional information than what is provided by the sensors, allowing our *deformer* method to be used on other datasets than nuScenes.<br>
Finally we also propose the first camera noise recognition method to recognize noise levels under the four different types of deformation we simulate, along with the first radar point cloud noise recognition model.

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
pip install open3d pandas
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
```shell
pip install jupyter pytorch-lightning wandb torchviz 
```


## Data preparation
Download and extract the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).<br>
Make sure your folder architecture corresponds to:
```
SynthCRAV
├──data
|   ├──default_nuScenes
|   |   ├── maps
|   |   ├── samples
|   |   ├── sweeps
|   |   ├── v1.0-mini
```
We advise using the synthesizer on the **mini dataset** only as the generated data takes about 40 times more space than the original data. If you want to use the trainval instead, download it and name it v1.0-trainval. Then make sure to set the ```split``` argument to ```--split train```.<br>
Make sure the naming of the dataset matches: **default_nuScenes**.

## Simulate and Recognize Different Levels of Noise 
### Dataset Generation
For each sensor, we generate 10 different noise levels, ranging from 10% to 100%, with a step size of 10%. The corrresponding folders are numbered 1 to 10. The original data is considered to be 0% noise, and thus is numbered 0. <br>
Given we simmulated four different degradations on images, the noise levels are generated for each type of noise.<br> 

To start generating the synthetic dataset:
```bash
python dataset_handler.py 
```
Warning: This assumes all previous steps have been completed as described.

The output architecture is as follows:
```
SynthCRAV
├──data
|   ├──default_nuScenes
|   |   ├── maps
|   |   ├── samples
|   |   ├── sweeps
|   |   ├── v1.0-mini
|   ├──noisy_nuScenes
|   |   ├── samples
|   |   |	├── CAM_[]
|   |   |	|	├── <noise_level>
|	|   |   |	|	├── <noise_type>
|   |   |	├── RADAR_[]
|   |   |	|	├── <noise_level>
```
Note we do not use the sweeps in this section, since the samples provide us with plenty enough data for the noise recognition task. However to generate a complete synthetically degraded dataset, we do---and recommend---using both samples and sweeps, to avoid complications with detectin backbones. 

### Noise Recognition Models
Once the noise synthetizer has finished, you can train or test our noise recognition models.<br>
Alternatively, find pre-trained checkpoints in ckpt/.

**train/test**
```bash
python noise_classifier.py
python noise_classifier.py
```

