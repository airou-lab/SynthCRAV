#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os
from moviepy import ImageSequenceClip
import matplotlib.pyplot as plt
import numpy as np

# Parameters
examples_root = './data/noisy_nuScenes/examples/presentation'

synth_folders = [item for item in os.listdir(examples_root) if os.path.isdir(os.path.join(examples_root,item))]
synth_folders.remove('default')
default_folders = list(os.listdir(os.path.join(examples_root,'default')))


print('examples_root:',examples_root)
print('synth_folders:',synth_folders)
print('default_folders:',default_folders)

for folder in default_folders:
    path = os.path.join(examples_root,'default',folder)
    image_list = [os.path.join(path,img_name) for img_name in os.listdir(path)]
    nfiles = len(image_list)

    image_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))

    print(image_list)

    output_file = os.path.join(examples_root,'default_'+folder+'.avi')
    
    clip = ImageSequenceClip(image_list, fps=2)
    clip.write_videofile(output_file, codec="libx264", fps=2, ffmpeg_params=["-crf", "18"])


for folder in synth_folders:
    path = os.path.join(examples_root,folder)
    image_list = [os.path.join(path,img_name) for img_name in os.listdir(path)]
    nfiles = len(image_list)

    image_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))

    print(image_list)

    output_file = os.path.join(examples_root,folder+'.avi')

    clip = ImageSequenceClip(image_list, fps=2)
    clip.write_videofile(output_file, codec="libx264", fps=2, ffmpeg_params=["-crf", "18"])


