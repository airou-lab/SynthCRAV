# SynthCRAV: CRN frontier experiments
## Getting Started
### Docker
Creating Docker image and container for this backbone
```shell
# Getting to Docker folder
cd ~/Documents/SynthCRAV/CRN_frontier/CRN/Docker

# Building CRN image (this can take a while)
sudo docker build -t crn .

# Creating mounted gpu-enabled GUI-enabled container (input this for every new shell)
xhost local:root	

## Validation dataset
sudo docker run --name CRN_frontier -v ~/Documents/SynthCRAV/CRN_frontier/CRN:/home/ws --gpus all --shm-size 16G -it \
            -v ~/Documents/SynthCRAV/data:/home/ws/data\
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
## Experiments
To perform the noise frontier experiments:
```bash
python ./data/CRN_frontier_exp.py 
```

To observe the resulting performances in mAP and NDS:
```bash
python ./data/viz_CRN_frontier.py 
```
