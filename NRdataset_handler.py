#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import numpy as np
import pandas as pd
import open3d as o3d
import copy
import argparse
import cv2
from math import cos, sin, asin, acos, atan2, sqrt, radians, degrees, pi, log

# Some useful functions
from utils.utils import *
from utils.fisheye import generate_fisheye_dist
from utils.synthesizer import deform_data

from visualizer import *
import visualizer

pd.set_option('display.max_rows', None)

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

# Dataset Parser (using kf because the other data aren't annotated and are interpolated)
def parse_nusc_keyframes(nusc, sensors, args, deformer):
    for scene in nusc.scene:
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('\nscene:\n',scene)

        while True:
            # Extract sensor data
            for sensor in sensors :
                # Load nusc info
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = os.path.join(args.nusc_root,sample_data['filename'])            
                
                if not args.debug:
                    print(150*' ',end='\r',flush=True)
                    print('current file:',filename,end='\r',flush=True)

                # Setting up output folder
                newfoldername = os.path.join(args.out_root,'samples', sensor, str(int(deformer.noise_level_radar*100)))
                mkdir_if_missing(newfoldername)

                if args.verbose:
                    print('nusc_sample:\n',nusc_sample)
                    print('sample_data:\n',sample_data)
                    print('Output folder name:',newfoldername)


                ##RADAR DATA SYNTHESIZER##
                if 'RADAR' in sensor:
                    # set output pcd file name
                    newfilename = os.path.join(newfoldername,filename.split('/')[-1])

                    # get current ego vel in sensor frame
                    deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)
                    
                    if args.verbose: 
                        print('output filename:',newfilename)
                        print('ego_vel:',deformer.ego_vel)

                    radar_df,_ = decode_pcd_file(filename,args.verbose)
                    
                    if radar_df.isna().any().any():
                        # Empty original radar point cloud check
                        print(150*'',end='\r')# clear print
                        print('NaN value in dataframe: skipped')
                        encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
                        continue

                    if args.gen_lvl_grad_img or args.gen_csv or args.gen_paper_img:
                        
                        if args.gen_lvl_grad_img:
                            noise_lvl_grad_gen_radar(args,filename,sensor,deformer,radar_df)

                        if args.gen_csv:
                            sample_name = filename.split('/')[-1].split('.')[0]
                            mkdir_if_missing('./noisy_nuScenes/examples/RADAR/csv/')
                            radar_df.to_csv('./noisy_nuScenes/examples/RADAR/csv/'+sample_name+'.csv')
                    
                        if args.gen_paper_img and sensor=='RADAR_FRONT':
                            gen_paper_img_radar(args, filename, sensor, deformer, radar_df)
                        continue

                    # Apply deformation
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    if len(deformed_radar_df) ==0:
                        # Empty generated radar point cloud check
                        print(150*'',end='\r')# clear print
                        print('Empty dataframe generated')
                                                 # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                                 # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
                        deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

                    # Save output
                    encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)

                    if args.disp_radar or args.save_radar_render:
                        # Read datapoints
                        dat = o3d.io.read_point_cloud(filename)
                        newdat = o3d.io.read_point_cloud(newfilename)

                        # converting to numpy format
                        pts_OG = np.asarray(dat.points)
                        pts_new = np.asarray(newdat.points)

                        if args.disp_radar:
                            # TODO: clean up a bit
                            disp_radar_pts(pts_OG,title='original',display=True, store_path='')
                            disp_radar_pts(pts_new,title='new',display=True, store_path='')
                            
                            # # using open3d built in (no axis)
                            # print('Original point clound')
                            # # print(np.asarray(dat.points))
                            # # print(dat)
                            # o3d.visualization.draw_geometries([dat])

                            # print('New point clound')
                            # # print(np.asarray(newdat.points))
                            # # print(newdat)
                            # o3d.visualization.draw_geometries([newdat])

                            # dat = o3d.io.read_point_cloud(filename)
                            # viz_radar_dat(sample_data)
                            # o3d.visualization.draw_geometries([dat])

                        if args.save_radar_render:
                            save_radar_3D_render(args,filename,pts_OG,pts_new,dat,newdat)


                ##CAMERA DATA SYNTHESIZER##
                elif 'CAM' in sensor:
                    if args.gen_lvl_grad_img or args.gen_paper_img:
                        # Loading image from filename        
                        img = cv2.imread(filename)

                        if args.gen_lvl_grad_img:
                            noise_lvl_grad_gen_cam(deformer,img,filename)
                        
                        if args.gen_paper_img and  sensor == 'CAM_FRONT':
                            gen_paper_img_cam(args, filename, sensor, deformer, radar_df)
                        continue

                    for deform_type in ['Blur','High_exposure','Low_exposure','Gaussian_noise']:
                        # Loading image from filename        
                        img = cv2.imread(filename)

                        # Output directory and file name
                        mkdir_if_missing(os.path.join(newfoldername,deform_type))
                        newfilename = os.path.join(newfoldername,deform_type,filename.split('/')[-1])
                        if args.verbose: print('output filename:',newfilename)


                        if 'night' in scene['description'].lower() and deform_type in ['High_exposure','Low_exposure']:
                            cv2.imwrite(newfilename, img)
                            # for data that has already low exposure and gaussian noise due to nighttime, only apply blur or noise, not other deform
                            continue                        

                        # Apply deformation
                        deformed_img = deformer.deform_image(img,deform_type)

                        # Save output
                        cv2.imwrite(newfilename, deformed_img)
                
                # next sensor
                if args.debug:
                    input('PRESS ANY KEY FOR NEXT SENSOR')

            if nusc_sample['next'] == "":
                #GOTO next scene
                print("no next data in scene %s"%(scene['name']))
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)


# auto-generation wrapper
def gen_dataset(nusc, sensors, args):
    '''
    Generating all noise levels for both sensors
    For Radars the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<name.pcd>
    For Cameras the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<noise_type>/<name.jpg>
    Resulting data is 50 x bigger than original dataset    
    '''

    for noise_level in range(10,110,10):
        deformer=deform_data(args)
        deformer.noise_level_radar = noise_level/100
        deformer.noise_level_cam = noise_level/100
        deformer.update_val()

        print(50*'-','Generating data at %d %% noise'%(noise_level),50*'-')

        parse_nusc_keyframes(nusc, sensors, args, deformer)

#--------------------------------------------------------------------Main--------------------------------------------------------------------
def create_parser():

    parser = argparse.ArgumentParser()
    
    # nuScenes loading
    parser.add_argument('--nusc_root', type=str, default='./data/default_nuScenes/', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='mini', help='train/val/test/mini')
    parser.add_argument('--sensor', type=str, nargs='+', default=sensor_list, help='Sensor type (see sensor_list) to focus on')

    # Noise level
    parser.add_argument('--n_level_cam', '-ncam','-cnoise' , type=float, default=0.1, help='Noise level for cams')
    parser.add_argument('--n_level_radar', '-nrad','-rnoise' , type=float, default=0.1, help='Noise level for radars')

    # Output config
    parser.add_argument('--out_root', type=str, default='./data/noisy_nuScenes', help='Noisy output folder')
    parser.add_argument('--no_overwrite', action='store_true', default=False, help='Do not overwrite existing output')

    # Display
    parser.add_argument('--disp_all_data', action='store_true', default=False, help='Display mosaic with camera and radar original info')
    parser.add_argument('--disp_radar', action='store_true', default=False, help='Display original Radar point cloud and new one')
    parser.add_argument('--save_radar_render', action='store_true', default=False, help='Save new original and new Radar point cloud 3D rendering')
    parser.add_argument('--disp_img', action='store_true', default=False, help='Display original Camera image and new one')
    parser.add_argument('--disp_all_img', action='store_true', default=False, help='Display mosaic of camera views')
    parser.add_argument('--gen_lvl_grad_img', action='store_true', default=False, help='generate output files for multiple noise levels')
    parser.add_argument('--gen_paper_img', action='store_true', default=False, help='generate output files for paper (4 outputs)')
    parser.add_argument('--gen_csv', action='store_true', default=False, help='generate csv file out of df (debug)')
    
    # Verbosity level
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity on|off')

    # Other
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--checksum', action='store_true', default=False, help='checks encoding/decoding of files')

    return parser



def check_args(args):
    assert args.split in ['train','val','test','mini'], 'Wrong split type'

    if args.sensor:
        assert all(sensor in sensor_list for sensor in args.sensor), 'Unknown sensor selected: %s'%(args.sensor)  
   
    assert os.path.exists(args.nusc_root), 'Data folder at %s not found'%(args.nusc_root)

    if not os.path.exists(args.out_root):
        mkdir_if_missing(args.out_root)

    print(args)

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Loading nuScenes
    nusc = load_nusc(args.split,args.nusc_root)

    # load arguments for visualizer functions
    visualizer.init_var(nusc,args)

    # Dataset parser for debug
    if args.debug: 
        deformer=deform_data(args)
        parse_nusc_keyframes(nusc, args.sensor, args, deformer)

    # generate noisy dataset
    gen_dataset(nusc, args.sensor, args)

    exit('end of script')



'''
# Launch commands:

## debugging / step-by-step
python dataset_handler.py --debug --sensor <SENSOR> -v

## one-shot run
python dataset_handler.py




#Some reading :
https://github.com/nutonomy/nuscenes-devkit/blob/05d05b3c994fb3c17b6643016d9f622a001c7275/python-sdk/nuscenes/utils/data_classes.py#L315
https://forum.nuscenes.org/t/detail-about-radar-data/173/5
https://forum.nuscenes.org/t/radar-vx-vy-and-vx-comp-vy-comp/283/4
https://conti-engineering.com/wp-content/uploads/2020/02/ARS-408-21_EN_HS-1.pdf

## Some papers on the impact of SNR on accuracy :
https://ieeexplore.ieee.org/abstract/document/55565
https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/610920#:~:text=The%20transmitted%20power%20has%20an,target%20%5B7%2C%208%5D.

## Some paper on adverse image distortion:
https://github.com/Gil-Mor/iFish
https://github.com/noahzn/FoHIS
'''
