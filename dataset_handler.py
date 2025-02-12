#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import sys 
import json 
import numpy as np
import pandas as pd
import open3d as o3d
import math
import random
import pcl
import struct
import copy
import argparse

from shutil import copyfile
from pyquaternion import Quaternion
from math import cos, sin
from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt

# load nuScenes libraries
from nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.data_classes import TrackingConfig

# Some utsefull functions
from utils import *

pd.set_option('display.max_rows', None)

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']  





# idea to explore : render_pointcloud_in_image and render it on the relevant cam (all except cam_back)


# Visualization and rendering functions
def render_radar_data(sample_data_token: str,axes_limit: float = 40,ax: plt.Axes = None,nsweeps: int = 1,underlay_map: bool = True,use_flat_vehicle_coordinates: bool = True):
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        """
        # Get sensor modality.
        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']        
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = nusc.get('sample_data', ref_sd_token)

        # Get aggregated radar point cloud in reference frame.
        # The point cloud is transformed to the reference frame for visualization purposes.
        pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

        # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
        # point cloud.
        radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        velocities = pc.points[8:10, :]  # Compensated velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
        velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
        velocities[2, :] = np.zeros(pc.points.shape[1])

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                 'otherwise the location does not correspond to the map!'
            NuScenesExplorer(nusc).render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        

        point_scale = 3.0
        scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show velocities.
        points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
        deltas_vel = points_vel - points
        deltas_vel = 6 * deltas_vel  # Arbitrary scaling
        max_delta = 20
        deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
        colors_rgba = scatter.to_rgba(colors)
        for i in range(points.shape[1]):
            ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)


        return ax

def viz_radar_dat(sample_data):
    # Visualize radar point clouds of this sample for this a specific sensor   

    fig, ax = plt.subplots(1,1)
    
    ax=render_radar_data(sample_data['token'], nsweeps=5, underlay_map=True, ax=ax) 

    ax.axis('off')

    plt.tight_layout()
    plt.show()

def viz_all_sample_img(nusc_sample, save=False):
    # Visualize mosaic of all Camera images of this sample   
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()
    
    for i, sensor in enumerate(cam_list):
        sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
        filename = 'nuScenes/'+sample_data['filename']

        img = plt.imread(os.path.join('nuScenes',sample_data['filename']))
        ax[i].imshow(img)

        ax[i].set_title(sensor)

    plt.show(block=False)

    figname = filename.split('/')[-1]
    plt.savefig('./noisy_nuScenes/samples/cam_img/'+figname)
    plt.close()

def viz_all_sample_radar(nusc_sample):
    # Visualize mosaic of all radar point clouds of this sample   
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()
    
    for i, sensor in enumerate(radar_list):
        sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
        filename = 'nuScenes/'+sample_data['filename']

        ax[i]=render_radar_data(sample_data['token'], nsweeps=5, underlay_map=True, ax=ax[i]) 

        ax[i].set_title(sensor)

    # unused axis
    ax[5].axis('off')


    plt.show()

def disp_all_sensor_mosaic(nusc_sample):
    # Visualize mosaic cof all sensors
    fig, ax = plt.subplots(4,3)
    
    ax = ax.flatten()

    for i, sensor in enumerate(sensor_list):
        sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
        if 'CAM' in sensor_list[i]:
            img = plt.imread(os.path.join('nuScenes',sample_data['filename'])) 
            ax[i].imshow(img)

        if 'RADAR' in sensor_list[i]:
            ax[i]=render_radar_data(sample_data['token'], nsweeps=5, underlay_map=True, ax=ax[i]) 

        ax[i].set_title(sensor_list[i])
    
    # unused axis
    ax[11].axis('off')


    plt.tight_layout()
    plt.show()
      
def disp_radar_pts(points,title='',display=True,save_path=''):

    # x = points['x'].to_numpy()
    # y = points['y'].to_numpy()
    # z = points['z'].to_numpy()
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    # Create the figure and axes object
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(x, y, z, c='blue', marker='o')

    # Set labels for the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # for i in range(0,360,10):
        # print(i)
    ax.view_init(elev=40, azim=180)
        # plt.savefig('test_'+str(i)) 

    # Set title for the plot
    ax.set_title(title)

    # Display the plot
    if display:
        plt.show()

    if save_path!='':
        plt.savefig(save_path) 

def disp_img_plt(imgs=[],rows=10,cols=10,title='',legends=[],block=True, save_path=''):
    # Display any given image in a matplotlib plot with automatic subplot sizing
    # if not imgs:
    #     pass

    # if len(imgs)==1:
    #     cols=1
    # else:
    #     cols=2

    # if len(imgs)%2 == 0:
    #     # len is pair
    #     rows = int(len(imgs)/2)
    # else:
    #     rows = int((len(imgs)+1)/2)

    if not imgs or len(imgs)>(rows*cols):
        pass


    # margin=50 # pixels
    # spacing=35 # pixels
    # dpi=100. # dots per inch

    # w_im = np.shape(imgs[0])[1]
    # h_im = np.shape(imgs[0])[0]

    # n_imgs = len(imgs)

    # width = (n_imgs*w_im+2*margin+spacing)/dpi
    # height= (n_imgs*h_im+2*margin+spacing)/dpi

    # left = margin/dpi/width #axes ratio
    # bottom = margin/dpi/height
    # wspace = spacing/float(width)

    # fig, ax = plt.subplots(rows,cols)
    fig, ax = plt.subplots(rows,cols,figsize=(16, 9))
    # fig, ax = plt.subplots(rows,cols,figsize=(width,height),dpi=dpi)
    # fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
    #                 wspace=wspace, hspace=wspace)
    ax = ax.flatten()

    if len(imgs)<rows*cols:
        for k in range(len(imgs),int(rows*cols),1):
            ax[k].axis('off')

    for i,img in enumerate(imgs):
        # correct color scale as images are loaded by openCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax[i].imshow(img)

        if legends:
            ax[i].set_title(legends[i])

    if title != '':
        fig.suptitle(title)
    
    plt.tight_layout(pad=0)
    plt.show(block=block)

    if save_path!='':
        plt.savefig(save_path) 

def disp_img_cv2(img,title,block=True, save_path=''):
    # Display any given image in a cv2 plot
    cv2.imshow(title,img)
    
    if block:
        while True:
            # showing the image 
            cv2.imshow(title,img)
              
            # waiting using waitKey method 
            if cv2.waitKey(1) == ord("\r"):
                break

        # Close all windows
        cv2.destroyAllWindows()

    if save_path!='':
        cv2.imwrite(save_path,img)



# utils functions
def rot_x(theta):
    theta=math.radians(theta)
    return np.array([[1,          0,           0],
                     [0, cos(theta), -sin(theta)],
                     [0, sin(theta),  cos(theta)]])

def rot_y(theta):
    theta=math.radians(theta)
    return np.array([[ cos(theta), 0,  sin(theta)],
                     [          0, 1,           0],
                     [-sin(theta), 0,  cos(theta)]])

def rot_z(theta):
    theta=math.radians(theta)
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta),  cos(theta), 0],
                     [         0,           0, 1]])



# Dataset Parser (most likely using kf to be faster)
def parse_nusc_keyframes(nusc, sensors, args):

    deformer=deform_data(args)

    for scene in nusc.scene:
        print('scene:\n',scene)
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('nusc_sample:\n',nusc_sample)

        while True:
            # visualization
            if args.disp_all_data:
                disp_all_sensor_mosaic(nusc_sample)

            # Extract sensor data
            for sensor in sensors :
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = 'nuScenes/'+sample_data['filename']

                get_ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])

                print('sample_data:\n',sample_data)            

                print(get_ego_pose)
                print(cs_record)
                print(sensor_record)

                # input()

                newfoldername = os.path.join(args.out_root,filename.split('/')[1], sensor)
                print('Output folder name:',newfoldername)
                mkdir_if_missing(newfoldername)

                if args.disp_all_img:
                    viz_all_sample_img(nusc_sample,save=True)
                    continue

                if 'RADAR' in sensor:
                    newfilename = os.path.join(newfoldername,filename.split('/')[-1])
                    print(newfilename)

                    radar_df = decode_pcd_file(filename)

                    # print(radar_df)
                    # print(sum(radar_df['vx'])/len(radar_df))
                    # input()
                    
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    encode_to_pcd_file(deformed_radar_df,filename,newfilename)

                    if args.checksum:
                        print('checking encode/decode pipeline of new file')
                        print('original data:')
                        print(radar_df)
                        print(100*'-')
                        input()

                        print('extracting from test.pcd')
                        test_df = decode_pcd_file('test.pcd')
                        print(test_df)
                        print(100*'-')
                        input()                


                    # Read datapoints
                    dat = o3d.io.read_point_cloud(filename)
                    newdat = o3d.io.read_point_cloud(newfilename)

                    # Applying rotation to data points
                    # # nuScenes radar coordinates    -->        Carthesian coordinates
                    # #     z  x                                       y 
                    # #     | /                       -->              | 
                    # # y___|/                                        z|__x
                    # # x forward, y left, z up                       x right, y up, z forward

                    # dat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
                    # newdat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
                    # dat.rotate(rot_x(-60), center=(0, 0, 0)) #give it a 3d effect
                    # newdat.rotate(rot_x(-60), center=(0, 0, 0)) #give it a 3d effect

                    # converting to numpy format
                    pts_OG = np.asarray(dat.points)
                    pts_new = np.asarray(newdat.points)


                    if args.disp_radar:
                        disp_radar_pts(pts_OG,title='original',display=True, save_path='')
                        disp_radar_pts(pts_new,title='new',display=True, save_path='')
                        
                        # using open3d built in (no axis)
                        # print('Original point clound')
                        # print(np.asarray(dat.points))
                        # print(dat)
                        # o3d.visualization.draw_geometries([dat])

                        # print('New point clound')
                        # print(np.asarray(newdat.points))
                        # print(newdat)
                        # o3d.visualization.draw_geometries([newdat])

                        # dat = o3d.io.read_point_cloud(filename)
                        # viz_radar_dat(sample_data)
                        # o3d.visualization.draw_geometries([dat])

                    if args.save_radar:
                        # Save image by screenshot. Not a great way but didn't find anything better
                        mkdir_if_missing(newfoldername+'/imgs')
                        
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_OG.png')
                        disp_radar_pts(pts_OG,title='original',display=False, save_path=image_path)
                        
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_new.png')
                        disp_radar_pts(pts_new,title='new',display=False, save_path=image_path)



                        dat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
                        newdat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord

                        vis = o3d.visualization.Visualizer()

                        vis.create_window() 
                        vis.add_geometry(dat)
                        vis.poll_events()
                        vis.update_renderer()
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_OG_open3d.png')
                        vis.capture_screen_image(image_path)
                        vis.destroy_window()
                        
                        vis.create_window() 
                        vis.add_geometry(newdat)
                        vis.poll_events()
                        vis.update_renderer()
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_new_open3d.png')
                        vis.capture_screen_image(image_path)
                        vis.destroy_window()


                elif 'CAM' in sensor:
                    # img = cv2.imread(filename)
                    deformed_img = deformer.deform_image(filename)

            if nusc_sample['next'] == "":
                #GOTO next scene
                print("no next data in scene %s"%(scene['name']))
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)

def parse_nusc(nusc):
    
    for scene in nusc.scene:
        print('scene:\n',scene)
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        sample_data = nusc.get('sample_data', nusc_sample['data']['CAM_FRONT'])
        while True:
            print('nusc_sample:\n',nusc_sample)
            print('sample_data:\n',sample_data)
            input()


            if sample_data['next'] == "":
                #GOTO next scene
                print("no next data in scene %s"%(scene['name']))
                break
            else:
                #GOTO next sample
                next_token = sample_data['next']
                sample_data = nusc.get('sample_data', next_token)




# Radar data extractor
def decode_pcd_file(filename):
    # Extract sensor data
    print('Opening point cloud data at:', filename)

    # opening file    
    meta = []
    with open (filename, 'rb') as file:
        for line in file:
            line = line.strip().decode('utf-8')
            meta.append(line)                        

            if line.startswith('DATA'):
                break

        data_binary = file.read()

    #extracting headers
    fields = meta[2].split(' ')[1:]
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)                    
    
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
             'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
             'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # store in dataframe
    df = pd.DataFrame(points,columns=fields, dtype=object)

    return df

def encode_to_pcd_file(df, ogfilename, newfilename):
    """
    Encode a Pandas DataFrame into a .pcd file.
    
    :param df: Pandas DataFrame containing radar data.
    :param filename: Output .pcd file path.
    """
    print('df to convert:')
    print(df)

    print('Opening original point cloud data at:', ogfilename)

    # opening file    
    meta = []
    with open (ogfilename, 'rb') as file:
        for line in file:
            line = line.strip().decode('utf-8')
            meta.append(line)                        

            if line.startswith('DATA'):
                break

        OG_data_binary = file.read()
    
    OGbinary_len = len(OG_data_binary)

    #extracting headers
    fields = meta[2].split(' ')[1:]
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    width = len(df)  # Number of points of new df
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types) 

    # fix for padding issue
    expected_size = width * sum([int(size) for size in sizes]) 
    padding_len = OGbinary_len - expected_size

    print('fields',fields)
    print('sizes',sizes)
    print('types',types)
    print('width',width)
    print('height',height)
    print('data',data)
    print('feature_count',feature_count)
    print('OGbinary_len',OGbinary_len)
    print('expected_size',expected_size)
    print('padding_len',padding_len)
    print()
    
    # Mapping from pandas dtypes to PCD types
    dtype_map = {
        'float32': ('F', 4),
        'float64': ('F', 8),
        'int8': ('I', 1),
        'int16': ('I', 2),
        'int32': ('I', 4),
        'int64': ('I', 8),
        'uint8': ('U', 1),
        'uint16': ('U', 2),
        'uint32': ('U', 4),
        'uint64': ('U', 8),
    }

    htype_map = {'F': {2: 'float16', 4: 'float32', 8: 'float64'},
       'I': {1: 'int8', 2: 'int16', 4: 'int32', 8: 'int64'},
       'U': {1: 'uint8', 2: 'uint16', 4: 'uint32', 8: 'uint64'}}
    
    # Verifing point types
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        htype = htype_map[str(types[i])][int(sizes[i])]

        # rectifying dtype if mismatch
        if dtype != htype:
            df[col] = df[col].astype(htype)
            dtype = str(df[col].dtype)

        # print(dtype, htype)

    # Construct the PCD header
    header = ['# .PCD v0.7 - Point Cloud Data file format',
    'VERSION 0.7',
    'FIELDS ' + ' '.join(fields),
    'SIZE ' + ' '.join(sizes),
    'TYPE ' + ' '.join(types),
    'COUNT ' + ' '.join(['1'] * len(fields)),
    'WIDTH ' + str(width),
    'HEIGHT ' + str(height),
    'VIEWPOINT 0 0 0 1 0 0 0',
    'POINTS ' + str(width),
    'DATA binary']

    # Create binary data
    binary_data = bytearray()
    packing_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                   'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                   'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    
    for i in range(width):
        for col, t, s in zip(df.columns, types, sizes):
            binary_data.extend(struct.pack(packing_lut[t][int(s)], df[col].iloc[i]))

    if padding_len:
        binary_data.extend(b'\x00' * padding_len)

    # Write to file
    with open(newfilename, 'wb') as file:
        # Writing header
        for item in header:
            file.write((item + '\n').encode('utf-8'))

        # Writing data
        file.write(binary_data)

    print(f"PCD file saved to {newfilename}")



# WORK IN PROGRESS
class deform_data():
    def __init__(self, args):
        # NuScenes radar model : Continental ARS408-21, 76∼77GHz https://conti-engineering.com/wp-content/uploads/2020/02/ARS-408-21_EN_HS-1.pdf
        # -Distance
        # -- Range: 0.20 to 250m far range | 0.20 to 70m/100m at [0;±45]° near range | 0.2 to 20m at ±60° near range
        # -- Resolution: Up to 1.79 m far range, 0.39 m near range
        # -Velocity
        # --Range: -400 km/h to +200 km/h (-leaving objects | +approximation)   --> -111.11 m/s to 55.55 m/s
        # --Resolution: 0.37 km/h far field, 0.43 km/h near range               --> 0.103 m/s ff, 0.119 m/s nr
        # radar sensor bounding values
        
        # radar sensor bounding values of position
        self.radar_sensor_bounds = {'dist':{'range':{'min_range':0.2,
                                                     'far_range':250,
                                                     'near_mid_range':100, 
                                                     'near_short_range':20
                                                    },
                                            'ang':{'far_range':9, # long range beam is +/- 9 degrees
                                                    'near_mid_range':45,
                                                    'near_short_range':60
                                                  },
                                            'resolution':{'far_range':1.79,
                                                          'near_range':0.39
                                                         }
                                            },
                                    'vel':{'range':[-111.11, 55.5],
                                            'resolution':{'far_range':0.103,
                                                          'near_range':0.119
                                                        }
                                            }
                                    }
        self.args = args
        self.verbose = args.verbose

        self.noise_level_radar = args.n_level_radar
        self.noise_level_cam = args.n_level_cam
    
    #---------------------------------------------------------Radar functions---------------------------------------------------------
    def get_ego_vel(self, df):
        ego_vx = df['vx'] - df['vx_comp']
        ego_vy = df['vy'] - df['vy_comp']

        return np.mean(ego_vx.to_numpy()), np.mean(ego_vy.to_numpy())

    def within_bound(self, x, y, vx, vy):
        # -Distance
        # -- Range: 0.20 to 250m far range | 0.20 to 70m/100m at [0;±45]° near range | 0.2 to 20m at ±60° near range
        # -- Resolution: Up to 1.79 m far range, 0.39 m near range

        if self.verbose: print('Verifying bounds:')


        if x<self.radar_sensor_bounds['dist']['range']['min_range'] or x>self.radar_sensor_bounds['dist']['range']['far_range']:
            # too close | too far
            if self.verbose: print('Out of min|max bounds')
            return False

        if vx < self.radar_sensor_bounds['vel']['range'][0] or vy < self.radar_sensor_bounds['vel']['range'][0] \
        or vx > self.radar_sensor_bounds['vel']['range'][1] or vy > self.radar_sensor_bounds['vel']['range'][1]:
            # velocity range, rarely even remotely reached
            if self.verbose: print('Out of velocity bounds')
            return False

        else:
            alpha = math.degrees(math.atan2(y,x))   # calculating point angle from sensor in degrees 
            if self.verbose: print('Alpha:', alpha)

            if x<=self.radar_sensor_bounds['dist']['range']['near_short_range']: 
                if self.verbose: print('Point is in near short-range')
                if abs(alpha)>self.radar_sensor_bounds['dist']['ang']['near_short_range']:
                    if self.verbose: print('Point is OOB with angle:',alpha)
                    return False

            elif x<=self.radar_sensor_bounds['dist']['range']['near_mid_range']: 
                if self.verbose: print('Point is in near mid-range')
                if abs(alpha)>self.radar_sensor_bounds['dist']['range']['near_mid_range']:
                    if self.verbose: print('Point is OOB with angle:',alpha)
                    return False

            else:   # far range
                if self.verbose: print('Point is in far range')
                if abs(alpha)>self.radar_sensor_bounds['dist']['ang']['far_range']:
                    if self.verbose: print('Point is OOB with angle:',alpha)
                    return False
        
        # if passed all tests
        if self.verbose: print('Point is within bounds')
        return True

    def within_resolution(self, df, x, y):
        # setting range type
        if x > 100:
            rg = 'far_range'
        else:
            rg = 'near_range'

        # extracting x,y values from dataset
        df_x = df['x'].astype('float32').to_numpy()
        df_y = df['y'].astype('float32').to_numpy()
        pt_df = np.column_stack((df_x, df_y))
        
        # Define the new point
        new_pt = np.array((x,y))
        
        if self.verbose:
            print('Checking resolution with other points of point cloud')
            print('Point clouds:\n',pt_df)
            print('Proposed point:\n',new_pt)

            print('\npoint range:',rg)
            print('min resolution:', self.radar_sensor_bounds['dist']['resolution'][rg])

            print(pt_df.dtype)  # Should be a floating-point type like float32 or float64
            print(new_pt.dtype)  # Should also be float32 or float6
            

            print('distances with all other points:\n', np.linalg.norm(pt_df - new_pt, axis=1))

        # return False if some points are too close
        if np.any(np.linalg.norm(pt_df - new_pt, axis=1) < self.radar_sensor_bounds['dist']['resolution'][rg]):
            if self.verbose: print('Resolution test failed')
            return False
        else: 
            if self.verbose: print('Within resolution')
            return True

    def create_ghost_point(self, df, i):
        if self.verbose:
            print('Generating ghost point')
            print('Using row %d as template:'%(i))
            print('Original values: \tx:',df.loc[i,'x'],'\ty:', df.loc[i,'y'])

        # Initialiazing point with inadmissible values
        x_fake=0
        y_fake=0
        vx_fake=1000
        vy_fake=1000

        if self.verbose: print('Initial ghost point: \tx:',x_fake,'\ty:', y_fake, '\tx:',vx_fake,'\ty:', vy_fake)

        # Controlling if point is realistic
        while not (self.within_bound(x_fake,y_fake,vx_fake,vy_fake) and self.within_resolution(df, x_fake, y_fake)):
            # maybe wrap this in a function later
            x_shift = np.random.normal(0, self.radar_sensor_bounds['dist']['range']['far_range']/6)
            y_shift = np.random.normal(0, 50) # theoretical max value for y is y_max = 70*tan(45)~=115 => 3sigmq = 115 => sigma = 115/3 ~=40 
            x_fake = df.loc[i,'x'] + x_shift
            y_fake = df.loc[i,'y'] + y_shift

            # getting range for corresponding velocity resolution
            if x_fake > self.radar_sensor_bounds['dist']['range']['near_mid_range'] :
                rg = 'far_range'
            else:
                rg = 'near_range'

            # Calculating velocity shift corresponding to position shift
            vx_shift = self.radar_sensor_bounds['vel']['resolution'][rg] * x_shift
            vy_shift = self.radar_sensor_bounds['vel']['resolution'][rg] * y_shift

            vx_fake = df.loc[i,'vx'] + vx_shift
            vy_fake = df.loc[i,'vy'] + vy_shift


            if self.verbose: print('Proposed ghost point: \tx:',x_fake,'\ty:', y_fake, '\tvx:',vx_fake,'\tvy:', vy_fake)
        
        # extracting ego motion from point_cloud, using it to re-create motion compensated fake points
        vx_ego, vy_ego = self.get_ego_vel(df)
        vx_fake_comp = vx_fake - vx_ego
        vy_fake_comp = vy_fake - vy_ego

        if self.verbose:
            print('Original values: \tx:',df.loc[i,'x'], '\ty:', df.loc[i,'y'],\
                                    '\tvx:',df.loc[i,'vx'], '\tvy:', df.loc[i,'vy'],\
                                    '\tvx_comp:',df.loc[i,'vx_comp'], '\tvx_comp:', df.loc[i,'vx_comp'])
            print('Retained points: \tx:',x_fake,'\ty:', y_fake, \
                                    '\tvx:',vx_fake,'\tvy:', vy_fake,\
                                    '\tvx_comp:',vx_fake_comp, '\tvx_comp:', vy_fake_comp)

        return x_fake, y_fake, vx_fake, vy_fake, vx_fake_comp, vy_fake_comp
            
    def FP_FN_gen(self, radar_df, noise_level):
        # generates rm_split % fake points and removes fake_spit % points from the dataframe

        # Initializing new dataframes and variables
        subset_df = copy.deepcopy(radar_df)
        ghost_df = pd.DataFrame(columns=subset_df.columns)
        n_rows=len(radar_df)

        # If no noise : return original dataset
        if noise_level == 0:
            return subset_df


        # Calculating chance of being dropped:
        # # Rule : 0% noise => 0%  chance of removal
        # #      100% noise => 75% chance of removal
        drop_rate = 0.75*noise_level # we still want to keep points even at 100% noise


        # Calculating chance of creating ghost point (chance for each point to be used to create a fake point, not necesserally outlier):
        # # Rule : 0% noise => 0%  chance of ghost
        # #      100% noise => 10% chance of ghost
        ghost_rate = 0.20*noise_level # Realistically ghost points remain pretty rare


        #-------------------------------------Ghost points generation-------------------------------------
        # Should we try to create clusters and outliers ? random gen should do that on its own but uncontrolled

        # Each row has a ghost_rate chance of being used to create a ghost point 
        random_vals = np.random.rand(n_rows)
        ghost_indices = np.where(random_vals <= ghost_rate)[0]
        print('ghost_indices:',ghost_indices)
        
        if ghost_indices.tolist():
            ghost_df = subset_df.iloc[ghost_indices]
            for i in ghost_indices:
                # Generating ghost points out of these real points
                x_fake, y_fake, vx_fake, vy_fake, vx_fake_comp, vy_fake_comp = self.create_ghost_point(subset_df, i)
                # Updating dataset values
                ghost_df.loc[i,['x','y','vx','vy','vx_comp','vy_comp']]= [x_fake, y_fake, vx_fake, vy_fake, vx_fake_comp, vy_fake_comp]
                # Recasting correct variable types
                ghost_df = ghost_df.astype({'x': 'float32', 'y': 'float32', 'vx': 'float32', 'vy': 'float32', 'vx_comp': 'float32', 'vy_comp': 'float32'}) 


        #----------------------------------------Random points drop----------------------------------------

        print(radar_df)

        # Each row has a drop_rate chance of being dropped 
        random_vals = np.random.rand(n_rows)
        drop_indices = np.where(random_vals <= drop_rate)[0]

        if self.verbose: 
            print('Removing %d random rows'%(len(drop_indices)))
            print('Removed rows:',drop_indices)

        subset_df = subset_df.drop(drop_indices, axis=0)
        subset_df.reset_index(drop=True, inplace=True)

        if self.verbose: print('Subset:', subset_df)

        return subset_df, ghost_df

    def gaussian_noise_gen(self, subset_df, noise_level): 
        # Generating n random points from a gaussian random distribution
        # noise_split is a percentage, the amount of points is this a subset of radar_df

        # we apply the noies uniformly to all point. Noise is a mix of gaussian and uniform rv
        # Initialization

        # subset_df = copy.deepcopy(df)

        print(subset_df)
        n_rows = len(subset_df)
        noise_arr = {'x':list(), 'y':list(),'vx':list(), 'vy':list(), 'type':list()}

   
        # --------------------------------Correlated position/velocity noise--------------------------------
        # (This is probably the most realisitic way to do it, with a joint noise model: v_noise = alpha x pos_noise)
        
        # Position
        ## generating n_rows noise values
        # Here different models

        # 80 % chance of gaussian noise, 9% of uniformely distributed, 1% of outlier
        
        noise_split_chance = np.random.rand(n_rows)
        noise_arr['type'] = ['outlier' if draw<0.01 else 'uniform' if draw<0.09 else 'gaussian' for draw in noise_split_chance]
        
        noise_arr['x'] = [  np.random.uniform(-self.radar_sensor_bounds['dist']['range']['far_range']/2, self.radar_sensor_bounds['dist']['range']['far_range']/2) if t == 'outlier' else
                            np.random.uniform(-5*noise_level, 5*noise_level) if 't' == 'uniform' else
                            10 * np.random.normal(0, noise_level) for t in noise_arr['type']]

        noise_arr['y'] = [  np.random.uniform(-self.radar_sensor_bounds['dist']['range']['far_range']/2, self.radar_sensor_bounds['dist']['range']['far_range']/2) if t == 'outlier' else
                            np.random.uniform(-5*noise_level, 5*noise_level) if 't' == 'uniform' else
                            10 * np.random.normal(0, noise_level) for t in noise_arr['type']]

        # Position Bounds security check
        for i in range(n_rows):
            # first checking if original point is inside of safety check
            if not(self.within_bound(subset_df.loc[i,'x'],subset_df.loc[i,'y'],0,0)):
                # If original point was OOB, make sure the noise gaussian (small) and skip
                noise_arr['x'][i] = np.random.normal(0, noise_level)
                noise_arr['y'][i] = np.random.normal(0, noise_level)
                pass
            else:
                x_new = subset_df.loc[i,'x'] + noise_arr['x'][i]
                y_new = subset_df.loc[i,'y'] + noise_arr['y'][i]

                if not (self.within_bound(x_new,y_new,0,0) and self.within_resolution(subset_df, x_new, y_new)):
                    while not (self.within_bound(x_new,y_new,0,0) and self.within_resolution(subset_df, x_new, y_new)):
                        t = noise_arr['type'][i]
                        if t == 'outlier':
                            noise_arr['x'][i] = np.random.uniform(-self.radar_sensor_bounds['dist']['range']['far_range']/2, self.radar_sensor_bounds['dist']['range']['far_range']/2)
                            noise_arr['y'][i] = np.random.uniform(-self.radar_sensor_bounds['dist']['range']['far_range']/2, self.radar_sensor_bounds['dist']['range']['far_range']/2)
                        elif t == 'uniform':
                            noise_arr['x'][i] = np.random.uniform(-5*noise_level, 5*noise_level)
                            noise_arr['y'][i] = np.random.uniform(-5*noise_level, 5*noise_level)
                        else:
                            noise_arr['x'][i] = 10 * np.random.normal(0, noise_level)
                            noise_arr['y'][i] = 10 * np.random.normal(0, noise_level)
                        x_new = subset_df.loc[i,'x'] + noise_arr['x'][i]
                        y_new = subset_df.loc[i,'y'] + noise_arr['y'][i]
        
        if self.verbose:
            print_df = pd.DataFrame()
            print_df['x'] = noise_arr['x'] 
            print_df['y'] = noise_arr['y'] 
            print_df['type'] = noise_arr['type'] 

            print(print_df)


        # converting to numpy array
        noise_arr['x']=np.array(noise_arr['x'])
        noise_arr['y']=np.array(noise_arr['y'])

        ## Previous method:
        # noise_arr['x'] = 10 * np.random.normal(0, noise_level, n_rows)
        # noise_arr['y'] = 10 * np.random.normal(0, noise_level, n_rows)
        # Deactivated due to OG points already being out sometimes 
        # # Position Bounds security check
        # for i in range(n_rows):
        #     x_new = subset_df.loc[i,'x'] + noise_arr['x'][i]
        #     y_new = subset_df.loc[i,'y'] + noise_arr['y'][i]

        #     if not (self.within_bound(x_new,y_new,0,0) and self.within_resolution(df, x_new, y_new)):
        #         while not (self.within_bound(x_new,y_new,0,0) and self.within_resolution(df, x_new, y_new)):
        #             noise_arr['x'][i] = 10 * np.random.normal(0, noise_level, n_rows)
        #             noise_arr['y'][i] = 10 * np.random.normal(0, noise_level, n_rows)
        #             x_new = subset_df.loc[i,'x'] + noise_arr['x'][i]
        #             y_new = subset_df.loc[i,'y'] + noise_arr['y'][i]


        ## Adding noise on x and y values
        subset_df.loc[:,'x']  += noise_arr['x']
        subset_df.loc[:,'y']  += noise_arr['y']


        if self.verbose: 
            print('resulting noise arrays:\n', noise_arr['x'])


        # Velocity 
        ## Getting range type for each points
        rg_arr = ['far_range' if noise_arr['x'][i]>self.radar_sensor_bounds['dist']['range']['near_mid_range'] else 'near_range' for i in range(n_rows)]

        ## Calculating velocity shift corresponding to position shift
        noise_arr['vx'] = [self.radar_sensor_bounds['vel']['resolution'][rg] for rg in rg_arr] * noise_arr['x']
        noise_arr['vy'] = [self.radar_sensor_bounds['vel']['resolution'][rg] for rg in rg_arr] * noise_arr['y']
        
        ## Adding noise on vx and vy values
        subset_df.loc[:,'vx']  += noise_arr['vx']
        subset_df.loc[:,'vy']  += noise_arr['vy']

        ## Corresponding noise on vx_comp and vy_comp
        vx_ego, vy_ego = self.get_ego_vel(subset_df)        # extracting ego motion from point_cloud, using it to re-create motion compensated noisy point
        subset_df.loc[:,'vx_comp'] = subset_df.loc[:,'vx'] - vx_ego
        subset_df.loc[:,'vy_comp'] = subset_df.loc[:,'vy'] - vy_ego

        # --------------------------------------------------------------------------------------------------

        return subset_df


    #--------------------------------------------------------Camera functions--------------------------------------------------------
    def blur(self,img):

        blur_level = 10*self.noise_level_cam
        ksize = max(3, int(2 * round(blur_level) + 1))
        sigma = blur_level 

        output_img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
        return output_img

    def high_exposure(self,img):
        # creating high exposure with a gaussian kernel x 2 (gotta figure out why this happens)
        gauss_kernel = (1/16) * np.array([[1,2,1],
                                          [2,4,2],
                                          [1,2,1]])

        # noise @ 10%  => 130% exposure (+30%)
        # noise @ 50%  => 250% exposure (+150%)
        # noise @ 100% => 400% exposure (+300%)
        kernel=gauss_kernel*(1+3*self.noise_level_cam)

        output_img = cv2.filter2D(src=img,ddepth=-1,kernel=kernel)

        return output_img

    def low_exposure(self,img):
        # creating low exposure with a gaussian kernel / 2 (gotta figure out why this happens)
        gauss_kernel = (1/16) * np.array([[1,2,1],
                                          [2,4,2],
                                          [1,2,1]])

        kernel=gauss_kernel/(1+3*self.noise_level_cam)

        output_img = cv2.filter2D(src=img,ddepth=-1,kernel=kernel)

        return output_img

    def add_noise(self,img):
        w = np.random.normal(0,self.noise_level_cam,img.shape).astype('uint8')

        output_img = cv2.add(img,w)

        return output_img


    def add_fog(self,img):


        # a promising git repo :
        # https://github.com/noahzn/FoHIS

        disp_img_cv2(image_fog, title='fog test', block=True)


        output_img = img
        return output_img


    #-------------------------------------------------Sensor-specific main functions-------------------------------------------------
    def deform_radar(self,radar_df):
        # NuScenes radar model : Continental ARS408-21, 76∼77GHz
        # -Distance
        # -- Range: 0.20 to 250m far range | 0.20 to 70m/100m at [0;±45]° near range | 0.2 to 20m at ±60° near range
        # -- Resolution: Up to 1.79 m far range, 0.39 m near range
        # -Velocity
        # --Range: -400 km/h to +200 km/h (-leaving objects | +approximation)   --> -111.11 m/s to 55.55 m/s
        # --Resolution: 0.37 km/h far field, 0.43 km/h near range               --> 0.103 m/s ff, 0.119 m/s nr

        # Noise level:
        # 0 means nothing changes (i.e output dataframe = input dataframe)
        # 100 % means : - 90 % of data is missing 
        #               - 10 % of ghost points generated
        #               - position and velocity Gaussian RVs can go from min to max theoretical values
        # 
        # Missing data can go from 0 to 90 of the dataset
        # Ghost point generation goes from 0 to 20 % of dataset size (and need to be spaced out of other points from at least the resolution value)
        # Position and velocity random variables are between 0 and their maximum values


        print('Original dataframe:')
        print(radar_df)
        
        for row in range(len(radar_df)):
            val = self.within_bound(radar_df.loc[row,'x'],radar_df.loc[row,'y'],radar_df.loc[row,'vx'],radar_df.loc[row,'vy'])
            print(val)
            if not val:
                print(radar_df.iloc[row])
                # input()

        # noise level dial (0.0 - 1.0)
        noise_level = self.noise_level_radar

        # Randomly removing and generating points
        trunc_df, ghost_df = self.FP_FN_gen(radar_df, noise_level)

        print('truncated dataframe:\n',trunc_df)
        print('ghost dataframe:\n',ghost_df)

        # Adding noise on remaining points (non-generated)
        noisy_df = self.gaussian_noise_gen(trunc_df, noise_level)

        if self.verbose:
            compare_df = pd.DataFrame(columns=['x_OG','x_new','y_OG','y_new','vx_OG','vx_new','vy_OG','vy_new','vx_comp_OG','vx_comp_new','vy_comp_OG','vy_comp_new'])
            compare_df[['x_OG','y_OG','vx_OG','vy_OG','vx_comp_OG','vy_comp_OG']] = trunc_df [['x','y','vx','vy','vx_comp','vy_comp']]
        
            compare_df[['x_new','y_new','vx_new','vy_new','vx_comp_new','vy_comp_new']] = noisy_df [['x','y','vx','vy','vx_comp','vy_comp']]

            print('Comparing original (subset) | noisy dataset:\n',compare_df)


        # Adding ghost points to noisy dataset
        final_df = pd.concat([noisy_df, ghost_df], axis=0, join='outer', ignore_index=True)

        # casting variable types on final dataset (even though this is re-done by the encoding file, this acts as a sort of security)
        final_df = final_df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32', 'vx_comp': 'float32', 'vy_comp': 'float32'}) 


        # Sorting dataset by id to shuffle in ghost points
        final_df.sort_values('id',axis=0,inplace=True)
        # Resetting index
        final_df.reset_index(drop=True, inplace=True)
        # Resetting id column to avoid doubles
        final_df['id'] = final_df.index 

        if self.verbose:
            print('Output dataframe:')
            print(final_df)

        return final_df

        
    def deform_image(self,filename):
        img = cv2.imread(filename)
        blur_img = copy.deepcopy(img)
        highexp_img = copy.deepcopy(img)
        lowexp_img = copy.deepcopy(img)
        noisy_img = copy.deepcopy(img)

        img_list =[]
        legends = []


        blur_img = self.blur(blur_img)
        highexp_img = self.high_exposure(highexp_img)
        lowexp_img = self.low_exposure(lowexp_img)
        noisy_img=self.add_noise(noisy_img)

        foggy_img = self.add_fog(img)



        #----------------display each type at current noise level in one plot-----------------

        # disp_img_plt([img], title='original', block=True)
        # disp_img_cv2(img, title='original', block=False)
        # img_list.append(img)
        # legends.append('original')

        # # Blurring image
        # blur_img = self.blur(blur_img)
        # # disp_img_cv2(blur_img, title='blurred', block=False)
        # img_list.append(blur_img)
        # legends.append('blurry')

        # # High exposure image
        # highexp_img = self.high_exposure(highexp_img)
        # # disp_img_cv2(highexp_img, title='highexp-noise', block=False)
        # img_list.append(highexp_img)
        # legends.append('high exposure')

        # # Low exposure image
        # lowexp_img = self.low_exposure(lowexp_img)
        # # disp_img_cv2(lowexp_img, title='lowexp-noise', block=True)
        # img_list.append(lowexp_img)
        # legends.append('low exposure')
        # disp_img_plt(imgs=img_list,title='Cluster',legends=legends,block=True)


        #--------------------display a plot of each type at all noise levels----------------------
        # img_list =[]
        # legends = []
        # img_list.append(img)
        # legends.append('original')
        # for i in range(1,11,1):
        #     blur_img = copy.deepcopy(img)
        #     self.noise_level_cam=i/10
        #     blur_img = self.blur(blur_img)
        #     img_list.append(blur_img)
        #     legends.append('lvl: '+str(i/10))
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='blur levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='blur levels tests',legends=legends,block=False,save_path='image_tests/blur.png')

        # img_list =[]
        # legends = []
        # img_list.append(img)
        # legends.append('original')
        # for i in range(1,11,1):
        #     highexp_img = copy.deepcopy(img)
        #     self.noise_level_cam=i/10
        #     highexp_img = self.high_exposure(highexp_img)
        #     img_list.append(highexp_img)
        #     legends.append('lvl: '+str(i/10))
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,,title='high exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='high exposure levels tests',legends=legends,block=False,save_path='image_tests/highexp.png')

        # img_list =[]
        # legends = []
        # img_list.append(img)
        # legends.append('original')
        # for i in range(1,11,1):
        #     lowexp_img = copy.deepcopy(img)
        #     self.noise_level_cam=i/10
        #     lowexp_img = self.low_exposure(lowexp_img)
        #     img_list.append(lowexp_img)
        #     legends.append('lvl: '+str(i/10))
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=False,save_path='image_tests/lowexp.png')

        # img_list =[]
        # legends = []
        # img_list.append(img)
        # legends.append('original')
        # for i in range(10):
        #     noisy_img = copy.deepcopy(img)
        #     self.noise_level_cam=(i+1)/10
        #     noisy_img = self.add_noise(noisy_img)
        #     img_list.append(noisy_img)
        #     legends.append('lvl: '+str(self.noise_level_cam))
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Gaussian noise levels tests',legends=legends,block=False,save_path='image_tests/Gauss_noise.png')


        exit()






#--------------------------------------------------------------------Main--------------------------------------------------------------------
def create_parser():

    parser = argparse.ArgumentParser()
    
    # nuScenes loading
    parser.add_argument('--nusc_root', type=str, default='./nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='mini', help='train/val/test/mini')
    parser.add_argument('--sensor', type=str, default=None, help='Sensor type (see sensor_list) to focus on')
    parser.add_argument('--at_scene', type=str, default=None, help='Select specific scene to drop in')
    parser.add_argument('--keyframes', '-kf', action='store_true', default=False, help='Only use keyframes (no sweeps, 2Hz instead of 12)')

    # Noise level
    parser.add_argument('--n_level_cam', '-ncam', type=float, default=0.1, help='Noise level for cams')
    parser.add_argument('--n_level_radar', '-nrad', type=float, default=0.1, help='Noise level for radars')

    # Output config
    parser.add_argument('--out_root', type=str, default='./noisy_nuScenes', help='Noisy output folder')

    # Display
    parser.add_argument('--disp_all_data', action='store_true', default=False, help='Display mosaic with camera and radar original info')
    parser.add_argument('--disp_radar', action='store_true', default=False, help='Display original Radar point cloud and new one')
    parser.add_argument('--save_radar', action='store_true', default=False, help='Save screenshot of original Radar point cloud and new one')
    parser.add_argument('--disp_img', action='store_true', default=False, help='Display original Camera image and new one')
    parser.add_argument('--disp_all_img', action='store_true', default=False, help='Display mosaic of camera views')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbosity on|off')

    # Other
    parser.add_argument('--debug', action='store_true', default=False, help='Debug argument')
    parser.add_argument('--checksum', action='store_true', default=False, help='checks encoding/decoding of files')






    return parser

def check_args(args):
    sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']  

    assert args.split in ['train','val','test','mini'], 'Wrong split type'
    assert args.sensor in sensor_list, 'Unknown sensor selected'    
   
    assert os.path.exists(args.nusc_root), 'Data folder at %s not found'%(args.nusc_root)

    print(args)

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    if args.sensor:
        sensors=[args.sensor]
    else:
        sensors=sensor_list

    # Loading scenes
    nusc = load_nusc(args.split,args.nusc_root)

    # Dataset parser
    parse_nusc_keyframes(nusc, sensors, args)

    exit('end of script')
