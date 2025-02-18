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
from math import cos, sin, asin, acos, atan2, sqrt
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

# Some useful functions
from utils.utils import *
from utils.fisheye import generate_fisheye_dist

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
      
def disp_radar_pts(points,title='',display=True,store_path=''):

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

    if store_path!='':
        plt.savefig(store_path) 

def disp_img_plt(imgs=[],rows=10,cols=10,title='',legends=[],block=True, store_path=''):
    # Display any given image in a matplotlib plot with automatic subplot sizing

    if not imgs or len(imgs)>(rows*cols):
        pass

    fig, ax = plt.subplots(rows,cols,figsize=(16, 9))
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

    if store_path!='':
        plt.savefig(store_path) 

def disp_img_cv2(img,title,block=True, store_path=''):
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

    if store_path!='':
        cv2.imwrite(store_path,img)



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

def polar_to_cart(r,theta):

    x = r * cos(theta)
    y = r * sin(theta)

    return x,y

def cart_to_polar(x,y):

    theta = atan2(y,x)
    r = sqrt(x**2 + y**2)

    return r, theta




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

                    print(radar_df)
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
                        disp_radar_pts(pts_OG,title='original',display=True, store_path='')
                        disp_radar_pts(pts_new,title='new',display=True, store_path='')
                        
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
                        # Saving radar point cloud rendering in matplotlib and in open3d bird eyeview
                        mkdir_if_missing(newfoldername+'/imgs')
                        
                        # Saving in matplotlib with auto angling for 3d effect
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_plt_OG.png')
                        disp_radar_pts(pts_OG,title='original',display=False, store_path=image_path)
                        
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_plt_new.png')
                        disp_radar_pts(pts_new,title='new',display=False, store_path=image_path)


                        # Save image by screenshot. Not a great way but apartently the only one 
                        dat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
                        newdat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord

                        vis = o3d.visualization.Visualizer()

                        vis.create_window() 
                        vis.add_geometry(dat)
                        vis.poll_events()
                        vis.update_renderer()
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_o3d_OG.png')
                        vis.capture_screen_image(image_path)
                        vis.destroy_window()
                        
                        vis.create_window() 
                        vis.add_geometry(newdat)
                        vis.poll_events()
                        vis.update_renderer()
                        image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_o3d_new.png')
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
        # 
        # near range encompasses short and mid range
        
        # radar sensor bounding values of position
        self.radar_sensor_bounds = {'dist':{'range':{'min_range':0.2,
                                                     'short_range':20,
                                                     'mid_range':100, 
                                                     'far_range':250
                                                     
                                                    },
                                            'ang':{'short_range':60,
                                                   'mid_range':45,
                                                   'far_range':9   # long range beam is +/- 9 degrees                                                    
                                                  },

                                            'resolution':{'near_range':0.39,
                                                          'far_range':1.79
                                                         }
                                            },
                                    'vel':{'range':[-111.11, 55.5],
                                            'resolution':{'near_range':0.119,
                                                          'far_range':0.103
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

            if x<=self.radar_sensor_bounds['dist']['range']['short_range']: 
                if self.verbose: print('Point is in near short-range')
                if abs(alpha)>self.radar_sensor_bounds['dist']['ang']['short_range']:
                    if self.verbose: print('Point is OOB with angle:',alpha)
                    return False

            elif x<=self.radar_sensor_bounds['dist']['range']['mid_range']: 
                if self.verbose: print('Point is in near mid-range')
                if abs(alpha)>self.radar_sensor_bounds['dist']['range']['mid_range']:
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

    def create_ghost_point(self, num_ghosts, radar_df):
        # Generating fake points
        # To better simulate the reception of points the generation is made in polar coordinates
        # Velocity is generated in cartesian coordinate as we don't have enough information to generate it in polar coordinates.
        # For more realistic velocities we sample from the current objects' and compensate from ego velocity reconstruction 
        ghost_df = pd.DataFrame(columns=radar_df.columns)

        for i in range(num_ghosts):


            #---- Generating x,y,z coordinates ------

            # no need to check bounds as they are guaranteed by the uniform distribution
            r = np.random.uniform(0.2,250)

            if r<=self.radar_sensor_bounds['dist']['range']['short_range']:    # short range
                print('point in short range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['short_range'],self.radar_sensor_bounds['dist']['ang']['short_range'])

            elif r<=self.radar_sensor_bounds['dist']['range']['mid_range']:    # mid range
                print('point in mid range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['mid_range'],self.radar_sensor_bounds['dist']['ang']['mid_range'])

            else: # far range
                print('point in far range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['far_range'],self.radar_sensor_bounds['dist']['ang']['far_range'])


            x, y = polar_to_cart(r,theta)

            print('r_fake:',r)
            print('theta_fake:',theta)
            print('x_fake:',x)
            print('y_fake:',y)

            
            #---- Generating dynamic properties ------
            dyn_prop = np.random.randint(0,8) # possible values are 0 to 7 included

            # dynProp: Dynamic property of cluster to indicate if is moving or not.
            # 0: moving
            # 1: stationary
            # 2: oncoming
            # 3: stationary candidate
            # 4: unknown
            # 5: crossing stationary
            # 6: crossing moving
            # 7: stopped

            if dyn_prop in [1, 3, 5, 7]:
                # stationary objects
                dyn = 'stationary'

            elif dyn_prop == 0:
                # moving but not incoming
                dyn = 'moving'

            elif dyn_prop == 2:
                # oncoming (-x direction)
                dyn = 'oncoming'

            elif dyn_prop == 4:
                # unknown
                dyn = 'unknown'

            elif dyn_prop == 6:
                # crossing moving (lateral moving)
                dyn = 'cross_moving'

            print('point is:',dyn)

            #---- Generating dynamic properties ------
            # TODO instead of drawing from possible dyn prop draw from current out of dataset, then take properties out of similar dyn objs
            print(radar_df[radar_df['dyn_prop'].isin([1, 3, 5, 7])][['vx','vy']])

            # if dyn == 'stationary':


            exit()

            # if dyn == 'stationary':
            #     # sample from static objects
            #     static_list = radar_df[radar_df['']]

            # vx = np.random()

            # x  y  z  dyn_prop  id  rcs  vx  vy  vx_comp  vy_comp  is_quality_valid  ambig_state  x_rms  y_rms  invalid_state  pdh0  vx_rms  vy_rms

            rcs = 0
            vx=0
            vy=0
            vx_comp=0
            vy_comp=0
            x_rms=0
            y_rms=0
            invalid_state=0
            pdh0=0
            vx_rms=0
            vy_rms=0

            row = [x, y, 0.0, dyn_prop, 0, rcs, vx, vy, vx_comp, vy_comp, 1, 3, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms]

# https://github.com/nutonomy/nuscenes-devkit/blob/05d05b3c994fb3c17b6643016d9f622a001c7275/python-sdk/nuscenes/utils/data_classes.py#L315
# https://forum.nuscenes.org/t/detail-about-radar-data/173/5
# https://forum.nuscenes.org/t/radar-vx-vy-and-vx-comp-vy-comp/283/4



    def FP_FN_gen(self, radar_df):
        # Simulating ghost points and missed points

        # Initializing new dataframes and variables
        subset_df = copy.deepcopy(radar_df)
        ghost_df = pd.DataFrame(columns=subset_df.columns)
        n_rows=len(radar_df)

        # Calculating chance of being dropped:
        # # Rule : 0% noise => 0%  chance of removal
        # #      100% noise => 75% chance of removal
        drop_rate = 0.75*self.noise_level_radar # we still want to keep points even at 100% noise


        # Ghost points appearing rate is independant of the noise level
        # => fixed, very small rate
        ghost_rate = 3 # max amount of ghost points: 3


        #-------------------------------------Ghost points generation-------------------------------------
        # Should we try to create clusters and outliers ? random gen should do that on its own but uncontrolled

        # Randomly draws how many ghost points will appear in this sample from a uniform distribution U(0,ghost_rate+1)
        num_ghosts = int(100 * ghost_rate * np.random.randint(low=0,high=ghost_rate+1))

        #temporary for debug:
        num_ghosts = 1
        if self.verbose:
            print('Generating %d ghost point'%(num_ghosts))

        if num_ghosts:
            # Generating random ghost points
            ghost_df = self.create_ghost_point(num_ghosts, radar_df)
            # # Recasting correct variable types
            # ghost_df = ghost_df.astype({'x': 'float32', 'y': 'float32', 'vx': 'float32', 'vy': 'float32', 'vx_comp': 'float32', 'vy_comp': 'float32'}) 
        exit()

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
        rg_arr = ['far_range' if noise_arr['x'][i]>self.radar_sensor_bounds['dist']['range']['mid_range'] else 'near_range' for i in range(n_rows)]

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
        # Blurring images using a convolution with a gaussian kernel
        # Increasing noise level increases kernel size, from [(3x3),(21x21)]

        output_img = copy.deepcopy(img)

        blur_level =2 * round(10*self.noise_level_cam)
        ksize = max(3, int(blur_level + 1))
        sigma = blur_level 

        output_img = cv2.GaussianBlur(output_img, (ksize, ksize), sigma)
        
        return output_img

    def high_exposure(self,img):
        # Creating high exposure with a gaussian kernel x (1+3xn)
        # Increasing noise increases pixel intensity (n), intensity from [1,4]
        
        output_img = copy.deepcopy(img)

        gauss_kernel = (1/16) * np.array([[1,2,1],
                                          [2,4,2],
                                          [1,2,1]])

        # noise @ 10%  => 130% exposure (+30%)
        # noise @ 50%  => 250% exposure (+150%)
        # noise @ 100% => 400% exposure (+300%)
        kernel=gauss_kernel*(1+3*self.noise_level_cam)

        output_img = cv2.filter2D(src=output_img,ddepth=-1,kernel=kernel)

        return output_img

    def low_exposure(self,img):
        # Creating low exposure with a gaussian kernel x (1+3xn)
        # Increasing noise decreases pixel intensity (n), intensity from [1,1/4]

        output_img = copy.deepcopy(img)

        gauss_kernel = (1/16) * np.array([[1,2,1],
                                          [2,4,2],
                                          [1,2,1]])

        kernel=gauss_kernel/(1+3*self.noise_level_cam)

        output_img = cv2.filter2D(src=output_img,ddepth=-1,kernel=kernel)

        return output_img

    def add_noise(self,img):
        # Adding random zero mean Gaussian noise to image
        # Simulates thermal or EM noise
        # Increasing nois level increases noise std_dev (sigma) from [0,1]

        output_img = copy.deepcopy(img)
        w = np.random.normal(0,self.noise_level_cam,output_img.shape).astype('uint8')

        output_img = cv2.add(output_img,w)

        return output_img

    def superfish(self,img,auto_resize=False):
        # Creating superfish distortion using modified code from https://github.com/Gil-Mor/iFish
        # Distortion level from [0,1] (above causes image warping)

        output_img = copy.deepcopy(img)
        dist_level = self.noise_level_cam

        output_img = generate_fisheye_dist(output_img,dist_level,auto_resize)

        return output_img


    def add_fog(self,img):
        # TODO
        output_img = copy.deepcopy(img)


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

        if self.verbose:
            print('Original dataframe:')
            print(radar_df)

        # Randomly removing and generating points
        trunc_df, ghost_df = self.FP_FN_gen(radar_df)

        if self.verbose:
            print('truncated dataframe:\n',trunc_df)
            print('ghost dataframe:\n',ghost_df)

        # Adding noise on remaining points (non-generated)
        noisy_df = self.gaussian_noise_gen(trunc_df, self.noise_level_radar)

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
        # Initializaing variables
        img_list =[]
        legends = []
        
        # Loading image from filename        
        img = cv2.imread(filename)

        # Generating various noises
        blur_img = self.blur(img)
        highexp_img = self.high_exposure(img)
        lowexp_img = self.low_exposure(img)
        noisy_img=self.add_noise(img)

        superfish_img = self.superfish(img)



        # foggy_img = self.s(img)




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

        # # Noisy image
        # noisy_img = self.add_noise(img)
        # disp_img_cv2(noisy_img, title='gaussian-noise', block=True)

        # # Superfish image
        # superfish_img = self.superfish(img)
        # disp_img_cv2(superfish_img, title='superfish-distort', block=True)


        #--------------------display a plot of each type at all noise levels----------------------
        # # Blurring image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     self.noise_level_cam=(i+1)/10                                               # Convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.blur(img))                                             # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='blur levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='blur levels tests',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/blur.png')
        
        
        # # High exposure image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     self.noise_level_cam=(i+1)/10                                               # Convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.high_exposure(img))                                    # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,,title='high exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='high exposure levels tests',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/highexp.png')
        

        # # Low exposure image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     self.noise_level_cam=(i+1)/10                                               # convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.low_exposure(img))                                     # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/lowexp.png')

        
        # # Noisy image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     self.noise_level_cam=(i+1)/10                                               # Convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.add_noise(img))                                        # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='low exposure levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Gaussian noise levels tests',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/Gauss_noise.png')


        # # Superfish image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     self.noise_level_cam=(i+1)/10                                               # Convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.superfish(img))                                        # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Superfish super high distortion levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Superfish distortion levels tests',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/superfish.png')


        # # Superfish image
        # img_list =[img]
        # legends = ['original']
        # for i in range(10):
        #     if i==4:
        #         continue
        #     self.noise_level_cam=(i+1)/10                                               # Convert i to noise level
        #     print('generating noise level: %d'%(int(self.noise_level_cam*100)))
        #     img_list.append(self.superfish(img,True))                                   # Apply and store transform at level i/100
        #     legends.append('lvl: '+str(self.noise_level_cam))                           # Store level legend
        # # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Superfish super high distortion levels tests',legends=legends,block=True)
        # disp_img_plt(imgs=img_list,rows=3,cols=4,title='Superfish distortion levels tests with auto-resize',legends=legends,block=False,store_path='noisy_nuScenes/image_tests/superfish_resize.png')


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
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Verbosity on|off')

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
