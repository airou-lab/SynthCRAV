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

from shutil import copyfile
from pyquaternion import Quaternion

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



# unused stuff
def bin():
    # filename = 'nuScenes/'+sample_data['filename']
    # print('\nfilename:',filename)
    # print ('opening point cloud data')

    # dat = o3d.io.read_point_cloud(filename)
    # print('\nraw dat:\n',dat)
    # print('\nraw points:\n',dat.points)

    # print('\npoints:\n', np.asarray(dat.points))

    # # o3d.visualization.draw_geometries([dat])
    # # viz_radar_dat(sample_data)

    pass

def process_radar_dat(nusc):
    # for scene in nusc.scene:
    #     print('scene:\n',scene)
    #     nusc_sample = nusc.get('sample', scene['first_sample_token'])
    #     while True:
    #         sample_data_list = []
    #         print('nusc_sample:\n',nusc_sample)

    #         # Extract sensor data
    #         for sensor in radar_list :
    #             sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
    #             print('sample_data:\n',sample_data)

    #             filename = 'nuScenes/'+sample_data['filename']
    #             print('\nfilename:',filename)
    #             print('opening point cloud data')
    #             print(100*'-')
    #             #----------------------------------------------------------------------Testing Grounds----------------------------------------------------------------------
                
    #             meta = []
    #             with open (filename, 'rb') as file:
    #                 for line in file:
    #                     line = line.strip().decode('utf-8')
    #                     meta.append(line)                        

    #                     if line.startswith('DATA'):
    #                         break
    #                     # print(line)

    #                 data_binary = file.read()
    #                 # print(binary_dat)

    #             fields = meta[2].split(' ')[1:]
    #             sizes = meta[3].split(' ')[1:]
    #             types = meta[4].split(' ')[1:]
    #             width = int(meta[6].split(' ')[1])
    #             height = int(meta[7].split(' ')[1])
    #             data = meta[10].split(' ')[1]
    #             feature_count = len(types)                    
                
    #             unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
    #                      'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
    #                      'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    #             types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    #             # Decode each point.
    #             offset = 0
    #             point_count = width
    #             points = []
    #             for i in range(point_count):
    #                 point = []
    #                 for p in range(feature_count):
    #                     start_p = offset
    #                     end_p = start_p + int(sizes[p])
    #                     assert end_p < len(data_binary)
    #                     point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
    #                     point.append(point_p)
    #                     offset = end_p
    #                 points.append(point)

    #             print(points)

    #             print(fields)

    #             df = pd.DataFrame(points,columns=fields)

    #             print(df)

    #             # cloud = RadarPointCloud.from_file(file_name=filename)

    #             # print(cloud.)

    #             dat = o3d.io.read_point_cloud(filename)
    #             # print('\nraw dat:\n',dat)
    #             print('\nraw points:\n',dat.points)

    #             print('\npoints:\n', np.asarray(dat.points))

    #             # # cloud = pcl.load(filename)

    #             # # print(cloud)

    #             # exit()
    #             o3d.visualization.draw_geometries([dat])
    #             # viz_radar_dat(sample_data)


    #             #----------------------------------------------------------------------Testing Grounds----------------------------------------------------------------------
    #             exit()
                
    #         if nusc_sample['next'] == "":
    #             #GOTO next scene
    #             print("no next data in scene %s"%(scene['name']))
    #             break
    #         else:
    #             #GOTO next sample
    #             next_token = nusc_sample['next']
    #             nusc_sample = nusc.get('sample', next_token)
    pass

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
    fig, ax = plt.subplots(1,1)
    
    ax=render_radar_data(sample_data['token'], nsweeps=5, underlay_map=True, ax=ax) 

    ax.axis('off')

    plt.tight_layout()
    plt.show()

def disp_sensor_dat(sample_data_list):
    fig, ax = plt.subplots(4,3)
    
    ax = ax.flatten()

    for i, sample_data in enumerate(sample_data_list):
        if 'CAM' in sensor_list[i]:
            img = plt.imread(os.path.join('nuScenes',sample_data['filename'])) 
            ax[i].imshow(img)

        if 'RADAR' in sensor_list[i]:
            ax[i]=render_radar_data(sample_data['token'], nsweeps=5, underlay_map=True, ax=ax[i]) 

        ax[i].set_title(sensor_list[i])

    ax[11].axis('off')


    plt.tight_layout()
    plt.show()
      



# Dataset Parser (most likely using kf to be faster)
def parse_nusc_keyframes(nusc, sensor_list):

    deformer=deform_data()

    for scene in nusc.scene:
        print('scene:\n',scene)
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('nusc_sample:\n',nusc_sample)

        while True:
            sample_data_list = []

            # Extract sensor data
            for sensor in sensor_list :
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = 'nuScenes/'+sample_data['filename']

                print('sample_data:\n',sample_data)            

                if 'RADAR' in sensor :
                    newfilename = 'test.pcd'

                    radar_df = extract_radar_dat(filename)
                    deformed_radar_df = deformer.deform_radar(radar_df)
                    encode_pcd(deformed_radar_df,filename,newfilename)

                    print('original data:')
                    print(radar_df)
                    print(100*'-')

                    print('extracting from test.pcd')
                    test_df = extract_radar_dat('test.pcd')
                    print(test_df)
                    print(100*'-')                

                    dat = o3d.io.read_point_cloud(filename)
                    o3d.visualization.draw_geometries([dat])
                    print()
                    print(np.asarray(dat.points))
                    print(dat)
                    print()

                    newdat = o3d.io.read_point_cloud(newfilename)
                    print()                    
                    print(np.asarray(newdat.points))
                    print(newdat)
                    print()                    
                    o3d.visualization.draw_geometries([newdat])

                    # exit()

                    
                    exit()

                sample_data_list.append(sample_data)


            # visualization
            if disp:
                disp_sensor_dat(sample_data_list)
                
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
def extract_radar_dat(filename):
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

def encode_pcd(df, ogfilename, newfilename):
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
    def __init__(self):
        pass
    
    @classmethod
    def gen_index_list(cls, size, n):
        # generate a list of random index numbers
        index_list = []
        while len(index_list) != n:
            index = random.randint(0, size-1)
            if index not in index_list:
                index_list.append(index)

        return index_list


    def gaussian_noise_gen(self, radar_df, mean=0, std=1, noise_split=1, verbose=True):    # should this not be noise level applied to all points ? Yes, set to 1 by default
        # Generating n random points from a gaussian random distribution
        # noise_split is a percentage, the amount of points is this a subset of radar_df

        # output df
        noisy_df = copy.deepcopy(radar_df)


        # setting amount of modified points
        n_mod_pts = int(noise_split*len(radar_df))

        # generating n_mod_pts noise values
        noise_arr = {'x':list(), 'y':list()}
        noise_arr['x']=np.random.normal(mean, std, n_mod_pts)
        noise_arr['y']=np.random.normal(mean, std, n_mod_pts)
        
        if verbose:
            print('Generating %d random points' %(n_mod_pts))
            print(noise_arr)

        # selecting random points to add noise to
        index_list = self.gen_index_list(size=len(radar_df),n=n_mod_pts)
        
        if verbose:
            print('randomly selecting %d rows' %(n_mod_pts))
            print(index_list)
            
        # adding noise on x and y values
        noisy_df.loc[index_list,'x'] += noise_arr['x']
        noisy_df.loc[index_list,'y'] += noise_arr['y']

        return noisy_df

        # TODO : also needs to be done to velocity values but they are smaller (ergo smaller noise)

    def FP_FN_gen(self, radar_df, rm_split=0.1, fake_split=0.1, verbose=True):
        # generates rm_split % fake points and removes fake_spit % points from the dataframe
        # NuScenes radar model : Continental ARS408-21, 76∼77GHz
        # -Distance
        # -- Range: 0.20 to 250m far range | 0.20 to 70m/100m at [0;±45]° near range | 0.2 to 20m at ±60° near range
        # -- Resolution: Up to 1.79 m far range, 0.39 m near range
        # -Velocity
        # --Range: -400 km/h to +200 km/h (-leaving objects | +approximation)   --> -111.11 m/s to 55.55 m/s
        # --Resolution: 0.37 km/h far field, 0.43 km/h near range               --> 0.103 m/s ff, 0.119 m/s nr


        subset_df = copy.deepcopy(radar_df)

        n_rm = int(rm_split*len(radar_df))
        n_fake = int(fake_split*len(radar_df))


        # removing n_rm random points
        if verbose:
            print('Removing %d random rows'%(n_rm))
        rm_index_list = self.gen_index_list(size=len(radar_df),n=n_rm)
        subset_df = subset_df.drop(rm_index_list, axis=0)
        subset_df.reset_index(drop=True, inplace=True)

        if verbose:
            print('Removed rows:',rm_index_list)

        # adding n-fake fake points
        ## We do this by copying points and adding random values to them (large gaussian variable to create outlisers as well as clusters)?
        if verbose:
            print('Creating %d randomly generated ghost points'%(n_fake))
        fake_index_list = self.gen_index_list(size=len(radar_df),n=n_fake)
        x_list = np.random.normal(0, 10, n_fake)
        y_list = np.random.normal(0, 10, n_fake)
        vx_list = np.random.normal(0, 1, n_fake)
        vy_list = np.random.normal(0, 1, n_fake)

        ghost_df = radar_df.loc[fake_index_list]
        if verbose:
            print('Using following rows:')
            print(ghost_df)

        ghost_df.loc[:,'x'] += x_list
        ghost_df.loc[:,'y'] += y_list
        ghost_df.loc[:,'vx_comp'] += vx_list
        ghost_df.loc[:,'vy_comp'] += vy_list
        
        if verbose:
            print('Ghost points:')
            print(ghost_df)

        output_df = pd.concat([subset_df, ghost_df], axis=0, join='outer', ignore_index=True)

        return subset_df, ghost_df


    def deform_radar(self,radar_df):
        print('Original dataframe:')
        print(radar_df)

        # Randomly removing and generating points
        trunc_df, ghost_df = self.FP_FN_gen(radar_df, rm_split=0.5, fake_split=0.1, verbose=False)

        # Adding noise on remaining points (non-generated)
        noisy_df = self.gaussian_noise_gen(trunc_df, mean=0, std=1, verbose=False)

        final_df = pd.concat([noisy_df, ghost_df], axis=0, join='outer', ignore_index=True)

        final_df = final_df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32', 'vx_comp': 'float32', 'vy_comp': 'float32'})

        print('Output dataframe:')
        print(final_df)

        return final_df

        
    def deform_image(self):
        pass


# Variables
split = 'mini'
keyframes = True
disp = False
# force-focus on specific sensor:
sensor = 'RADAR_FRONT'
# force-focus on specific scene:
at_scene = None

if __name__ == '__main__':

    sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

    cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

    radar_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']  

    if sensor:
        sensor_list=[sensor]



    # Loading scenes
    nusc = load_nusc(split,'nuScenes')

    # Dataset parser
    parse_nusc_keyframes(nusc, sensor_list)



    # process_radar_dat(nusc)

    # exit()


    # if keyframes:
    #     parse_nusc_keyframes(nusc)
    # else:
    #     parse_nusc(nusc)


    # exit(1)
