# MatTrack: Camera-Radar 3D Multi-Object Tracking, Camera-Radar Backbone (CRN)

## Getting Started

### Data preparation
If you have followed all the steps properly, you should only have to create the data folder:
```shell
cd ~/Documents/MatTrack/Detection/CRN
mkdir data
```

### Docker
Creating Docker image and container for this backbone
```shell
# Getting to Docker folder
cd ~/Documents/MatTrack/Detection/CRN/Docker

# Building CRN image (this can take a while)
sudo docker build -t crn .

# Creating mounted gpu-enabled GUI-enabled container (input this for every new shell)
xhost local:root	

## Validation dataset
sudo docker run --name CRN_frontier -v ~/Documents/MatData/CRN_frontier/CRN:/home/ws --gpus all --shm-size 16G -it \
            -v ~/Documents/MatData/data:/home/ws/data\
        	--env="DISPLAY" \
        	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        	crn:latest
```
You should now be in the container shell.

Upon encountering any memory-related issue, be sure to check the shared memory of the container using ```df -h```. A simple fix can be to increase the -shm-size.

### Installation
```shell
cd mmdetection3d
pip install -v -e .
cd ..

python setup.py develop  # GPU required

```
Make sure your nuscenes devkit is updated to the latest version :
```shell
pip install nuscenes-devkit -U
```


### Data preparation

**Step 1.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
*Note: this process requires test data, even for mini dataset*
```shell
python scripts/gen_info.py
```

**Step 2.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
*Note2: this process requires test data, even for mini dataset*
```shell
python scripts/gen_depth_gt.py
```

**Step 3.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```shell
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```

The folder structure will be as follows:
```
CRN
├── data
│   ├── nuScenes (link)
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
|   |   ├── depth_gt
|   |   ├── radar_bev_filter  # temporary folder, safe to delete
|   |   ├── radar_pv_filter
|   |   ├── v1.0-trainval

```

### Training and Evaluation
**Training**
```shell
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```shell
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```
*Example using R50* 
```shell
python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1
```
**Testing**  
Before testing, make sure you modify the script of ```scripts/gen_radar_bev.py and scripts/gen_radar_pv.py``` by uncommenting the testing blocs at the top of the files. <br>
Then, re-run step 4:
```shell
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```
And launch the test pipeline :
```shell
python [EXP_PATH] --ckpt_path [CKPT_PATH] -p -b 4 --gpus 4
```
Example with R50:
```shell
python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -p -b 1 --gpus 1
```


### Model Zoo
All models use 4 keyframes and are trained without CBGS.  
All latency numbers are measured with batch size 1, GPU warm-up, and FP16 precision.

|  Method  | Backbone | NDS  | mAP  | FPS  | Params | Config                                                  | Checkpoint                                                                                                  |
|:--------:|:--------:|:----:|:----:|:----:|:------:|:-------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| BEVDepth |   R50    | 47.1 | 36.7 | 29.7 | 77.6 M | [config](exps/det/BEVDepth_r50_256x704_128x128_4key.py) | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/BEVDepth_r50_256x704_128x128_4key.pth) |
|   CRN    |   R18    | 54.2 | 44.9 | 29.4 | 37.2 M | [config](exps/det/CRN_r18_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r18_256x704_128x128_4key.pth)      |
|   CRN    |   R50    | 56.2 | 47.3 | 22.7 | 61.4 M | [config](exps/det/CRN_r50_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r50_256x704_128x128_4key.pth)      |


### Features
- [ ] BEV segmentation checkpoints 
- [ ] BEV segmentation code 
- [x] 3D detection checkpoints 
- [x] 3D detection code 
- [x] Code release 

### Acknowledgement
This project is based on excellent open source projects:
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)


### Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{kim2023crn,
    title={Crn: Camera radar net for accurate, robust, efficient 3d perception},
    author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={17615--17626},
    year={2023}
}
```

## Prepare for Tracking (outside container)
Once CRN generated an output, copy the resulting data to the outputs folder:
```shell
# Trainval
cp outputs/det/CRN_r50_256x704_128x128_4key/* ../outputs/CRN_detection_output

# Test
cp outputs/det/CRN_r50_256x704_128x128_4key/* ../outputs/CRN_detection_output_test

# Mini
cp outputs/det/CRN_r50_256x704_128x128_4key/* ../outputs/CRN_detection_output_mini

```