#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

'''
Uses SynthCRAV deformer functions to recreate a dataset with simulated noises.
Creates a new mini and validation dataset by default. Can be used on train set as wlell but be careful of computation time.
'''

import os
import shutil
import random
import subprocess
import sys
import pandas as pd
import numpy as np
from argparse import Namespace
from utils.synthesizer import deform_data
from utils.utils import *

total_sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']


sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']


camera_degradation_types_list = ['Blur','Gaussian_noise','High_exposure','Low_exposure']

# Init args
args = Namespace(
# Noise level
n_level_cam = 0.1,
n_level_radar = 0.1,
# Verbosity level
verbose = 0,
#debug 
debug=False
)

def symlink_lidar(nusc_root, output_root):
    # symlink lidar data for compatibility
    if not os.path.exists(os.path.abspath(os.path.join(output_root,'samples','LIDAR_TOP'))):
        os.symlink(os.path.abspath(os.path.join(nusc_root,'samples','LIDAR_TOP')), os.path.abspath(os.path.join(output_root,'samples','LIDAR_TOP')))

    if not os.path.exists(os.path.abspath(os.path.join(output_root,'sweeps','LIDAR_TOP'))):
        os.symlink(os.path.abspath(os.path.join(nusc_root,'sweeps','LIDAR_TOP')), os.path.abspath(os.path.join(output_root,'sweeps','LIDAR_TOP')))

def get_number_of_samples(nusc, scene):
    # nusc_sample = nusc.get('sample', scene['first_sample_token'])
    # cnt = 0
    # while True:
    #     cnt+=1
    #     if nusc_sample['next'] == "":
    #         break
    #     else:
    #         nusc_sample = nusc.get('sample', nusc_sample['next'])
    
    # return cnt

    return scene['nbr_samples']

def random_noise_parser(nusc, scene, nusc_root, output_root):
    '''
    Random noise generation:
    @ each sample, decide which sensor type (CAM or RADAR) fails. Then, randomly draw a noise level (0 to 1, uniform distribution)
    deform the data and store the output file in output_root
    The other sensor type is directly copied to output_root without being changed
    '''
    # Initializing deformer class
    deformer=deform_data(args)
    deform_type = None
    deformer.noise_level_radar = 0
    deformer.noise_level_cam = 0
    deformer.update_val()

    # Initializing logger
    list_logger = []
    
    # Booting up nusc parser
    print('\n',scene['name'],':\n',scene)
    print('\n',40*'-','RANDOM NOISE GENERATION',40*'-','\n')
    nusc_sample = nusc.get('sample', scene['first_sample_token'])
    n_samples = get_number_of_samples(nusc, scene)
    cnt=0

    while True:
        # decide which sensor type to deform and at which level
        deformed_sensor = random.choice(['CAM','RADAR'])
        noise_lvl = round(np.random.randint(0,11)/10,2)

        if deformed_sensor == 'CAM':
            # select deformation type
            if 'night' in scene['description'].lower():
                # nighttime deformation cannot be exposure-related
                deform_type = random.choice(camera_degradation_types_list[:2])
            else:
                deform_type = random.choice(camera_degradation_types_list)
            
            # Update noise level in deformer class
            deformer.noise_level_cam = noise_lvl
            deformer.noise_level_radar = 0
            deformer.update_val()
        else:
            deform_type='N/A'
            # Update noise level in deformer class
            deformer.noise_level_cam = 0
            deformer.noise_level_radar = noise_lvl
            deformer.update_val()

        # Log noise info in logger for this token
        list_logger.append([scene['name'],nusc_sample['token'],deformed_sensor,noise_lvl,deform_type])



        # Extract sensor data
        for sensor in sensor_list:
            # Load nusc info
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
            filename = os.path.join(nusc_root,os.path.join(sample_data['filename']))
            print(200*' ',end='\r')  # clear print
            print('%s \t %d/%d'%(filename,cnt+1,n_samples),end='\r')

            # Setting up output folder
            newfilename =  os.path.join(output_root,sample_data['filename'])

            ##RADAR DATA SYNTHESIZER##
            if 'RADAR' in sensor:
                if deformed_sensor == 'RADAR':
                    # get current ego vel in sensor frame
                    deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)

                    # decode PCD file
                    radar_df,_ = decode_pcd_file(filename,args.verbose)
                    
                    # Empty original radar point cloud check
                    if radar_df.isna().any().any():
                        print('\nNaN value in dataframe: skipped')
                        encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
                        continue

                    # Apply deformation
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    # Empty resulting radar point cloud check
                    if len(deformed_radar_df) ==0:
                        print('\nEmpty dataframe generated')
                                                 # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                                 # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
                        deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

                    # Save output
                    encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)
                
                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)

            ##CAMERA DATA SYNTHESIZER##
            elif 'CAM' in sensor:
                if deformed_sensor == 'CAM':
                    # Loading image from filename
                    img = cv2.imread(filename)

                    # Apply deformation
                    deformed_img = deformer.deform_image(img,deform_type)

                    # Save output
                    cv2.imwrite(newfilename, deformed_img)

                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)


            if args.debug:
                print('Current sample:\n',nusc_sample)
                print('Sample token:',nusc_sample['token'])
                print()
                print('Deformed sensor:',deformed_sensor)
                print('Noise level:',noise_lvl)
                print()
                print('Deformer class radar noise level:',deformer.noise_level_radar)
                print('dB SNR decrease',deformer.SNR_decrease_dB)
                print('Linear SNR ratio',deformer.SNR_ratio_linear)
                print()
                print('Deformer class camera noise level:',deformer.noise_level_cam)
                print('Camera selected deformation type (if applicable):',deform_type)
                print()
                print('Original filename;',filename)
                print('New filename:',newfilename)
                print('Logger:',list_logger)
                print()
                input("NEXT SENSOR")

        if nusc_sample['next'] == "":
            #GOTO next scene
            print("\nno next data in scene %s"%(scene['name']))
            break
        else:
            #GOTO next sample
            cnt+=1
            next_token = nusc_sample['next']
            nusc_sample = nusc.get('sample', next_token)
    return list_logger

def ramp_up_parser(nusc, scene, nusc_root, output_root, deformed_sensor):
    '''
    Ramping up noise generation:
    For each scene a single sensor type (CAM or RADAR) fails. The noise level ramps up starting at 0 for the first few frams and ending at 100 for the last frames.
    The other sensor type is directly copied to output_root without being changed
    '''
    # Initializing deformer class
    deformer=deform_data(args)
    deform_type = None
    deformer.noise_level_radar = 0
    deformer.noise_level_cam = 0
    deformer.update_val()
    
    # only one deformation type per scene for cameras, not applicable to radar
    if deformed_sensor == 'CAM':
        # select deformation type
        if 'night' in scene['description'].lower():
            # nighttime deformation cannot be exposure-related
            deform_type = random.choice(camera_degradation_types_list[:2])
        else:
            deform_type = random.choice(camera_degradation_types_list)
    else:
        deform_type='N/A'

    # Initializing logger
    list_logger = []

    # Booting up nusc parser
    print('\n',scene['name'],':\n',scene)
    print('\n',40*'-','RAMP-UP NOISE GENERATION',40*'-')
    print(50*'-',deformed_sensor,50*'-','\n')
    nusc_sample = nusc.get('sample', scene['first_sample_token'])
    n_samples = get_number_of_samples(nusc, scene)
    cnt=0

    while True:
        # increment noise level based on token position
        noise_lvl = round(0.1*((cnt*10)//(0.9*n_samples)),1) # dividing by 0.9*n_samples to get final values at 1.0 instead of 0.9

        # Update noise level in deformer class
        if deformed_sensor == 'CAM':
            deformer.noise_level_cam = noise_lvl
            deformer.noise_level_radar = 0
        else:
            deformer.noise_level_cam = 0
            deformer.noise_level_radar = noise_lvl
        deformer.update_val()        

        # Log noise info in logger for this token
        list_logger.append([scene['name'],nusc_sample['token'],deformed_sensor,noise_lvl,deform_type])

        # Extract sensor data
        for sensor in sensor_list:
            # Load nusc info
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
            filename = os.path.join(nusc_root,os.path.join(sample_data['filename']))
            print(200*' ',end='\r')  # clear print
            print('%s \t %d/%d'%(filename,cnt+1,n_samples),end='\r')
            
            # Setting up output folder
            newfilename =  os.path.join(output_root,sample_data['filename'])

            ##RADAR DATA SYNTHESIZER##
            if 'RADAR' in sensor:
                if deformed_sensor == 'RADAR':
                    # get current ego vel in sensor frame
                    deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)

                    # decode PCD file
                    radar_df,_ = decode_pcd_file(filename,args.verbose)
                    
                    # Empty original radar point cloud check
                    if radar_df.isna().any().any():
                        print('\nNaN value in dataframe: skipped')
                        encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
                        continue

                    # Apply deformation
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    # Empty resulting radar point cloud check
                    if len(deformed_radar_df) ==0:
                        print('\nEmpty dataframe generated')
                                                 # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                                 # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
                        deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

                    # Save output
                    encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)
                
                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)

            ##CAMERA DATA SYNTHESIZER##
            elif 'CAM' in sensor:
                if deformed_sensor == 'CAM':
                    # Loading image from filename
                    img = cv2.imread(filename)

                    # Apply deformation
                    deformed_img = deformer.deform_image(img,deform_type)

                    # Save output
                    cv2.imwrite(newfilename, deformed_img)

                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)


            if args.debug:
                print('Current sample:\n',nusc_sample)
                print('Sample token:',nusc_sample['token'])
                print()
                print('Deformed sensor:',deformed_sensor)
                print('Noise level:',noise_lvl)
                print()
                print('Deformer class radar noise level:',deformer.noise_level_radar)
                print('dB SNR decrease',deformer.SNR_decrease_dB)
                print('Linear SNR ratio',deformer.SNR_ratio_linear)
                print()
                print('Deformer class camera noise level:',deformer.noise_level_cam)
                print('Camera selected deformation type (if applicable):',deform_type)
                print()
                print('Original filename;',filename)
                print('New filename:',newfilename)
                print('Logger:',list_logger)
                print()
                input("NEXT SENSOR")

        if nusc_sample['next'] == "":
            #GOTO next scene
            print("\nno next data in scene %s"%(scene['name']))
            break
        else:
            #GOTO next sample
            cnt+=1
            next_token = nusc_sample['next']
            nusc_sample = nusc.get('sample', next_token)

    return list_logger

def constant_parser(nusc, scene, nusc_root, output_root, deformed_sensor):
    '''
    Constant noise generation:
    For each scene a single sensor type (CAM or RADAR) fails. The noise level is at a constant value across all the samples.
    The other sensor type is directly copied to output_root without being changed
    '''
    # Initializing deformer class
    deformer=deform_data(args)
    deform_type = None
    deformer.noise_level_radar = 0
    deformer.noise_level_cam = 0
    deformer.update_val()
    
    # only one deformation type per scene for cameras, not applicable to radar
    if deformed_sensor == 'CAM':
        # select deformation type
        if 'night' in scene['description'].lower():
            # nighttime deformation cannot be exposure-related
            deform_type = random.choice(camera_degradation_types_list[:2])
        else:
            deform_type = random.choice(camera_degradation_types_list)
    else:
        deform_type='N/A'

    # Initializing logger
    list_logger = []

    noise_lvl = round(np.random.randint(3,11)/10,2) # considering higher noise levels for constant noise for it to have a use
    # Update noise level in deformer class  (requires only one update)
    if deformed_sensor == 'CAM':
        deformer.noise_level_cam = noise_lvl
        deformer.noise_level_radar = 0
    else:
        deformer.noise_level_cam = 0
        deformer.noise_level_radar = noise_lvl
    deformer.update_val()   
    
    # Booting up nusc parser
    print('\n',scene['name'],':\n',scene)
    print('\n',40*'-','CONSTANT NOISE GENERATION',40*'-')
    print(45*'-',deformed_sensor,noise_lvl,45*'-','\n')
    nusc_sample = nusc.get('sample', scene['first_sample_token'])
    n_samples = get_number_of_samples(nusc, scene)
    cnt=0

    while True:
        # Log noise info in logger for this token
        list_logger.append([scene['name'],nusc_sample['token'],deformed_sensor,noise_lvl,deform_type])

        # Extract sensor data
        for sensor in sensor_list:
            # Load nusc info
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
            filename = os.path.join(nusc_root,os.path.join(sample_data['filename']))
            print(200*' ',end='\r')  # clear print
            print('%s \t %d/%d'%(filename,cnt+1,n_samples),end='\r')
            
            # Setting up output folder
            newfilename =  os.path.join(output_root,sample_data['filename'])

            ##RADAR DATA SYNTHESIZER##
            if 'RADAR' in sensor:
                if deformed_sensor == 'RADAR':
                    # get current ego vel in sensor frame
                    deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)

                    # decode PCD file
                    radar_df,_ = decode_pcd_file(filename,args.verbose)
                    
                    # Empty original radar point cloud check
                    if radar_df.isna().any().any():
                        print('\nNaN value in dataframe: skipped')
                        encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
                        continue

                    # Apply deformation
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    # Empty resulting radar point cloud check
                    if len(deformed_radar_df) ==0:
                        print('\nEmpty dataframe generated')
                                                 # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                                 # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
                        deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

                    # Save output
                    encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)
                
                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)

            ##CAMERA DATA SYNTHESIZER##
            elif 'CAM' in sensor:
                if deformed_sensor == 'CAM':
                    # Loading image from filename
                    img = cv2.imread(filename)

                    # Apply deformation
                    deformed_img = deformer.deform_image(img,deform_type)

                    # Save output
                    cv2.imwrite(newfilename, deformed_img)

                # undeformed data is exact copy from original dataset
                else:
                    shutil.copy(filename, newfilename)


            if args.debug:
                print('Current sample:\n',nusc_sample)
                print('Sample token:',nusc_sample['token'])
                print()
                print('Deformed sensor:',deformed_sensor)
                print('Noise level:',noise_lvl)
                print()
                print('Deformer class radar noise level:',deformer.noise_level_radar)
                print('dB SNR decrease',deformer.SNR_decrease_dB)
                print('Linear SNR ratio',deformer.SNR_ratio_linear)
                print()
                print('Deformer class camera noise level:',deformer.noise_level_cam)
                print('Camera selected deformation type (if applicable):',deform_type)
                print()
                print('Original filename;',filename)
                print('New filename:',newfilename)
                print('Logger:',list_logger)
                print()
                input("NEXT SENSOR")

        if nusc_sample['next'] == "":
            #GOTO next scene
            print("\nno next data in scene %s"%(scene['name']))
            break
        else:
            #GOTO next sample
            cnt+=1
            next_token = nusc_sample['next']
            nusc_sample = nusc.get('sample', next_token)

    return list_logger

def unchanged_parser(nusc, scene, nusc_root, output_root):
    '''
    No deformation.
    Copies the original file to the synthetic folder
    '''
    # Booting up nusc parser
    print('\n',scene['name'],':\n',scene)
    print('\n',40*'-','NO NOISE - SIMPLE COPY',40*'-','\n')
    nusc_sample = nusc.get('sample', scene['first_sample_token'])
    n_samples = get_number_of_samples(nusc, scene)
    cnt=0


    while True:
        # Extract sensor data
        for sensor in sensor_list:
            # Load nusc info
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
            filename = os.path.join(nusc_root,os.path.join(sample_data['filename']))
            print(200*' ',end='\r')  # clear print
            print('%s \t %d/%d'%(filename,cnt+1,n_samples),end='\r')

            # Setting up output folder
            newfilename =  os.path.join(output_root,sample_data['filename'])

            shutil.copy(filename, newfilename)

            if args.debug:
                print('Current sample:\n',nusc_sample)
                print('Sample token:',nusc_sample['token'])
                print()
                print('Original filename;',filename)
                print('New filename:',newfilename)
                print()
                input("NEXT SENSOR")

        if nusc_sample['next'] == "":
            #GOTO next scene
            print("\nno next data in scene %s"%(scene['name']))
            break
        else:
            #GOTO next sample
            cnt+=1
            next_token = nusc_sample['next']
            nusc_sample = nusc.get('sample', next_token)


def apply_noise_single_file(deformer, filename, newfilename, nusc, nusc_sample, sensor, deformed_sensor, noise_level, deformation_type):
    if deformed_sensor == 'CAM':
        # update deformer
        deform_type = deformation_type
        deformer.noise_level_cam = noise_level
        deformer.noise_level_radar = 0
        deformer.update_val()

        # Loading image from filename
        img = cv2.imread(filename)

        # Apply deformation
        deformed_img = deformer.deform_image(img,deform_type)

        # Save output
        cv2.imwrite(newfilename, deformed_img)
    
    elif deformed_sensor == 'RADAR':
        # update deformer
        deform_type = None
        deformer.noise_level_cam = 0
        deformer.noise_level_radar = noise_level
        deformer.update_val()  

        # get current ego vel in sensor frame
        deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)

        # decode PCD file
        radar_df,_ = decode_pcd_file(filename,args.verbose)
        
        # Empty original radar point cloud check
        if radar_df.isna().any().any():
            print('\nNaN value in dataframe: skipped')
            encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
            return 0

        # Apply deformation
        deformed_radar_df = deformer.deform_radar(radar_df)

        # Empty resulting radar point cloud check
        if len(deformed_radar_df) ==0:
            print('\nEmpty dataframe generated')
                                     # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                     # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
            deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

        # Save output
        encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)

def sweeps_generator(nusc, nusc_root, output_root):
    '''
    Appliess deformation to sweeps.
    Sweeps deformation is the same as their corresponding sample
    '''

    # initialize deformer
    deformer=deform_data(args)
    deform_type = None
    deformer.noise_level_radar = 0
    deformer.noise_level_cam = 0
    deformer.update_val()

    go_flag = False

    #parse scenes
    for scene in nusc.scene:

        if scene['name'] == 'scene-0352':
            go_flag = True

        if not go_flag:
            print('passed', scene['name'])
            continue

        # extract temporal noise information (constant, random, ramp, unchanged, night)
        scene_split_file_path = os.path.join('datagen_logs',split,'scenes_split.txt')
        pickle_path = os.path.join('datagen_logs',split,scene['name']+'.pkl')
        
        assert os.path.exists(scene_split_file_path), 'Error, corrresponding .txt file not found for scene %s'%(scene['name'])
        assert os.path.exists(pickle_path), 'Error, corrresponding pickle file not found for scene %s'%(scene['name'])

        found_scene_flag = 0
        with open(scene_split_file_path,'r') as scene_split_file:
            for line in scene_split_file:
                line = line.strip()
                # print(line)
                if line.split(':')[0] == scene['name']:
                    found_scene_flag=1
                    scene_noise_type = line.split(' ')[1]
        
        assert found_scene_flag, 'Error parsing .txt file: %s not found'%(scene['name'])

        
        # extract detailed noise information (scene, token, deformed_sensor, noise_level, deformation_type) for each SAMPLE token of this scene
        scene_noise_detail = pd.read_pickle(pickle_path)
        
        print('scene name:',scene['name'])
        print('scene_noise_type:',scene_noise_type)
        print('scene_noise_detail:\n',scene_noise_detail)

        # parse all sensor info
        for sensor in sensor_list:
            print('CURRENT SENSOR:',sensor)

            # first sample and sweep extraction
            nusc_sample = nusc.get('sample', scene['first_sample_token'])
            sample_token = nusc_sample['token']
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])   # data for first sample
            sample_data_token = sample_data['token']

            while True:
                if args.debug:
                    print('nusc_sample:',nusc_sample)
                    print('sample_token:',sample_token)
                    print('sample_data:',sample_data)
                    print('sample_data_token:',sample_data_token)

                # only considering sweeps
                if not sample_data['is_key_frame']: 
                    # extract filenames
                    filename = os.path.join(nusc_root, sample_data['filename'])
                    newfilename = os.path.join(output_root, sample_data['filename'])

                    # applying noise
                    if scene_noise_type not in ['unchanged','night']:
                        token_noise_info = scene_noise_detail[scene_noise_detail['token'] == sample_token].iloc[0]

                        deformed_sensor = token_noise_info ['deformed_sensor']  # CAM or RADAR
                        noise_level = float(token_noise_info ['noise_level'])
                        deformation_type = token_noise_info ['deformation_type']

                        if args.debug:
                            print()
                            print('token_noise_info:\n',token_noise_info)
                            print()
                            print('deformed_sensor:',deformed_sensor)
                            print('noise_level:',noise_level)
                            print('deformation_type:',deformation_type)
                            print()
                            print('filename:',filename)
                            print('newfilename:',newfilename)

                        if deformed_sensor in sensor:
                            # apply noise to correct sensor
                            print(200*' ',end='\r')  # clear print
                            print('Deform:',filename,end='\r')
                            apply_noise_single_file(deformer, filename, newfilename, nusc, nusc_sample, sensor, deformed_sensor, noise_level, deformation_type)
                        else:
                            # other sensor is unchanged <=> copied
                            print(200*' ',end='\r')  # clear print
                            print('Copy:',filename,end='\r')
                            shutil.copy(filename, newfilename)
                    else:
                        # unchanged and night data are direct copies
                        print(200*' ',end='\r')  # clear print
                        print('Copy:',filename,end='\r')
                        shutil.copy(filename, newfilename)

                if args.debug:
                    input('NEXT DATA')
     
                if sample_data['next'] == "":
                    #GOTO next scene
                    print("\nno next data in scene %s for sensor %s"%(scene['name'],sensor))
                    break
                
                else:
                    #GOTO next sample (iterate on sample datas to get sweeps)
                    sample_data_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_data_token)
                    
                    # corresponding sample
                    sample_token = sample_data['sample_token']
                    sample = nusc.get('sample', sample_token)


def checksum(nusc, nusc_root, output_root):
    '''
    checks presence of all files in new dataset 
    '''

    #parse scenes
    for scene in nusc.scene:
        # parse all sensor info
        print('\n%s'%(scene['name']))
        for sensor in total_sensor_list:
            # first sample and sweep extraction
            nusc_sample = nusc.get('sample', scene['first_sample_token'])
            sample_token = nusc_sample['token']
            sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])   # data for first sample
            sample_data_token = sample_data['token']

            while True:
                # extract filenames
                filename = os.path.join(output_root, sample_data['filename'])
                print(150*' ',end='\r')
                print(filename,end='\r')

                if not os.path.exists(filename):
                    input('missing file: %s %s %s'%(scene, sample_token, filename))
     
                if sample_data['next'] == "":
                    #GOTO next scene
                    break
                
                else:
                    #GOTO next sample (iterate on sample datas to get sweeps)
                    sample_data_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_data_token)
                    
                    # corresponding sample
                    sample_token = sample_data['sample_token']
                    sample = nusc.get('sample', sample_token)
    print('Done.')



if __name__ == '__main__':
    # parameters
    nusc_root = './data/default_nuScenes'
    output_root = '../synth_nuScenes'
    log_root='./datagen_logs'
    
    # split = 'mini'
    split = 'val'

    assert os.path.exists(nusc_root), 'nusc_root not found at %s.'%(nusc_root)
    assert os.path.exists(output_root), 'output_root not found at %s.'%(output_root)

    # creating necessary output folders
    mkdir_if_missing(os.path.join(log_root,split))
    for sensor in sensor_list:
        mkdir_if_missing(os.path.join(output_root,'samples',sensor))
        mkdir_if_missing(os.path.join(output_root,'sweeps',sensor))

    symlink_lidar(nusc_root, output_root)
    
    nusc = load_nusc(split, nusc_root)


    # ------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------Split scenes---------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------
    # extract nuscenes scenes in a list
    scenes_list = [scene['name'] for scene in nusc.scene if 'night' not in scene['description'].lower()]
    night_scenes_list = [scene['name'] for scene in nusc.scene if 'night' in scene['description'].lower()]
    print('scenes list',scenes_list)
    print('nighttime scenes list',night_scenes_list)

    # shuffle scenes list
    random.shuffle(scenes_list)
    print('shuffled scenes list',scenes_list)


    # split in different temporal noise categories
    splits = {'random_noise':0.2,
                'ramp_up':0.3,
                'constant':0.3,
                'unchanged':0.2}
    print('splits:',splits)

    ## sanity check
    assert sum(splits.values()) == 1.0, 'Error: split percentages should be equal to 1.0'

    ## split
    split_sizes = [round(item*len(scenes_list)) for item in splits.values()]
    split_sizes[-1] = len(scenes_list) - sum(split_sizes[:-1])  # Ensure all scenes are used
    print(split_sizes)

    split_scenes = {key: list(group) for key, group in zip(splits.keys(), np.array_split(scenes_list, np.cumsum(split_sizes)[:-1]))}
    # not changing nighttime scenes
    split_scenes['night']=(night_scenes_list)
    print(split_scenes)

    scene_to_category = {scene: category for category, scenes in split_scenes.items() for scene in scenes}

    # log which scene gets which noise type in a txt file
    with open(os.path.join(log_root,split,'scenes_split.txt'),'w') as f:
        for scene_name, category in scene_to_category.items():
            f.write(f"{scene_name}: {category}\n")

    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------Apply deformations------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------
    for scene in nusc.scene:
        if scene['name'] in split_scenes['random_noise']:
            scene_logger = random_noise_parser(nusc, scene, nusc_root, output_root)

        elif scene['name'] in split_scenes['ramp_up']:
            # First 50% are CAM, second 50% are RADAR.
            # Since data is shuffled that assures both randomness and 50/50 split 
            if split_scenes['ramp_up'].index(scene['name']) < len(split_scenes['ramp_up'])//2:
                deformed_sensor='CAM'
            else:
                deformed_sensor='RADAR'
            scene_logger = ramp_up_parser(nusc, scene, nusc_root, output_root, deformed_sensor)

        elif scene['name'] in split_scenes['constant']:
            # First 50% are CAM, second 50% are RADAR.
            # Since data is shuffled that assures both randomness and 50/50 split 
            if split_scenes['constant'].index(scene['name']) < len(split_scenes['constant'])//2:
                deformed_sensor='CAM'
            else:
                deformed_sensor='RADAR'
            scene_logger = constant_parser(nusc, scene, nusc_root, output_root, deformed_sensor)

        elif scene['name'] in split_scenes['unchanged'] or scene['name'] in split_scenes['night']:
            # no modification here, parsing only copies files
            unchanged_parser(nusc, scene, nusc_root, output_root)
            scene_logger=[]

        else:
            print('something went wrong with the splits...')
            exit(-1)
        
        df_logger = pd.DataFrame(scene_logger,columns=['scene','token','deformed_sensor','noise_level','deformation_type'])
        df_logger.to_pickle(os.path.join(log_root,split,scene['name']+'.pkl'))


    sweeps_generator(nusc, nusc_root, output_root)

    checksum(nusc, nusc_root, output_root)

