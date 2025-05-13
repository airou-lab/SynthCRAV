#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import numpy as np
import pandas as pd
import open3d as o3d
import struct
import copy
import argparse
from shutil import copyfile
from pyquaternion import Quaternion
from math import cos, sin, asin, acos, atan2, sqrt, radians, degrees, pi, log
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nuscenes import NuScenesExplorer
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud

from utils.utils import *

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']



def viz_nusc(nusc,nusc_root,singleSensor='CAM_FRONT'):
    # parse nusc and display camera views
    for scene in nusc.scene:
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('scene:\n',scene)
        print('nusc_sample:\n',nusc_sample)
        while True:
            sample_data = nusc.get('sample_data', nusc_sample['data'][singleSensor])
            filename = os.path.join(nusc_root,sample_data['filename'])
            
            # viz_all_cam_img(nusc_sample)
            img = cv2.imread(filename)
            disp_img_cv2(img,scene['name'])

            if nusc_sample['next'] == "":
                #GOTO next scene
                print("no next data in scene %s"%(scene['name']))
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)

def render_radar_data(sample_data_token: str,axes_limit: float = 40,ax: plt.Axes = None,nsweeps: int = 1,underlay_map: bool = True,use_flat_vehicle_coordinates: bool = True, show_vel: bool = True):
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

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

        point_scale = 25.0
        scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)


        return ax

def render_radar_3D_scan(nusc, nusc_sample, n, savepath='',figname=''):
    # Create the figure and axes object
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    for radar in radar_list:
        sample_data = nusc.get('sample_data', nusc_sample['data'][radar])

        ax=render_radar_data(sample_data['token'], nsweeps=1, underlay_map=False, ax=ax, show_vel=False) 

    ax.view_init(elev=30, azim=-140)
    ax.axis('off')
    plt.tight_layout()
    
    if savepath=='':
        plt.show()
    else:
        plt.savefig(os.path.join(savepath,figname))

    plt.close()


if __name__ == '__main__':
    # nusc = load_nusc('mini', './data/default_nuScenes/')
    # nusc = load_nusc('mini', './data/disp_nuscenes/')
    # nlvl='100'

    # for scene in nusc.scene:
    #     nusc_sample = nusc.get('sample', scene['first_sample_token'])
    #     n=1
    #     while True:
    #         print(nusc_sample)

    #         render_radar_3D_scan(nusc, nusc_sample, n=n, savepath='./data/noisy_nuScenes/examples/better_radar_evolution', figname=nlvl)
    #         # render_radar_3D_scan(nusc, nusc_sample, n=n, savepath='')
            
    #         exit()

    #         if nusc_sample['next'] == "":
    #             #GOTO next scene
    #             # break
    #             exit()
    #         else:
    #             #GOTO next sample
    #             n+=1
    #             next_token = nusc_sample['next']
    #             nusc_sample = nusc.get('sample', next_token)

    # Visualize mosaic of radar point clouds
    filelist = ['0','30','60','100']
    fig, ax = plt.subplots(2,2,figsize=(60, 60))
    ax = ax.flatten()

    for i, file in enumerate(filelist):
        filename = './data/noisy_nuScenes/examples/better_radar_evolution/'+file+'.png'
        img = 255 - cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = plt.imread(filename)
        ax[i].imshow (img)
        ax[i].set_title(file.split('.')[0]+'% noise')

        ax[i].set_xticks([])
        ax[i].set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0.1)  # No width or height space
    # plt.tight_layout()
    plt.show()
