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
# import pcl
import struct
import copy
import argparse

from shutil import copyfile
from pyquaternion import Quaternion
from math import cos, sin, asin, acos, atan2, sqrt, radians, degrees, pi, log
# from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt

# load nuScenes libraries
from nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes

# Some useful functions
from utils.utils import *
from utils.fisheye import generate_fisheye_dist

pd.set_option('display.max_rows', None)

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']





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

def save_radar_pointcloud(newfoldername,filename,pts_OG,pts_new,dat,newdat):
    # Saving radar point cloud rendering in matplotlib and in open3d bird eyeview
    mkdir_if_missing(newfoldername+'/imgs')
    
    # Saving in matplotlib with auto angling for 3d effect
    image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_plt_OG.png')
    disp_radar_pts(pts_OG,title='original',display=False, store_path=image_path)
    
    image_path = os.path.join(newfoldername,'imgs',filename.split('/')[-1].split('.')[0].split('_')[-1]+'_plt_new.png')
    disp_radar_pts(pts_new,title='new',display=False, store_path=image_path)


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

def radar_df_to_excel(radar_df,filename):
    name = filename.split('/')[-1].split('.')[0]+'.xlsx'
    radar_df.to_excel   ('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+name)
    print('saved to %s'%('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+name))
    input()

def noise_lvl_grad_gen(args,filename,sensor,deformer,radar_df):
    # generate a gradient of noise levels in matplotlib subplot
    deformer.radar_ghost_max=0  # debug, TODO: remove

    token = filename.split('/')[-1].split('.')[0]
    output_folder = os.path.join(args.out_root,'samples',sensor,'noise_lvl',token)
    mkdir_if_missing(output_folder)
    vis = o3d.visualization.Visualizer()

    # extract original point cloud
    dat = o3d.io.read_point_cloud(filename)

    # Saving original pcd in matplotlib with auto angling for 3d effect
    pts_OG = copy.deepcopy(np.asarray(dat.points))
    pts_list=[pts_OG]
        
    # temp save of o3d visualization
    image_path = os.path.join(output_folder,'o3d_0.png')
    dat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
    vis.create_window() 
    vis.add_geometry(dat)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(image_path)

    for noise_lvl in range (1,11,1):
        df_og = copy.deepcopy(radar_df)
        deformer.noise_level_radar = (noise_lvl+1)/10
        print('noise level at %f %%'%(deformer.noise_level_radar * 100))
        newfilename = os.path.join(output_folder,token+'_'+str(noise_lvl)+'.pcd')
        
        deformed_radar_df = deformer.deform_radar(df_og)
        
        encode_to_pcd_file(deformed_radar_df,filename,newfilename)

        # Read datapoints
        newdat = o3d.io.read_point_cloud(newfilename)

        # Converting to numpy format
        pts_new = copy.deepcopy(np.asarray(newdat.points))
        pts_list.append(pts_new)

        # Temp save of o3d visualization
        image_path = os.path.join(output_folder,'o3d_'+str(noise_lvl)+'.png')
        newdat.rotate(rot_z(90), center=(0, 0, 0)) # correct rotation to carthesian coord
        vis.clear_geometries()
        vis.add_geometry(newdat)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path)

    vis.destroy_window()

    # Gather o3d images in a list
    imlist=[]
    name_list=[]
    for item in os.listdir(output_folder):
        # remove previously generated files
        if item[-3:]=='png':
            if (token in item) or ('comp' in item): 
                # previous grid generation
                os.remove(os.path.join(output_folder,item))
    
    for item in os.listdir(output_folder):
        # add newly generated files      
        if item[-3:]=='png' and item[:3]=='o3d':
            name_list.append(item)
    
    for i in range(len(name_list)):
        name = 'o3d_'+str(i)+'.png'
        imlist.append(cv2.imread(os.path.join(output_folder,name)))


    # Open3D reconstruction and saving in a subplot:
    fig, ax = plt.subplots(3,4,figsize=(16, 9))
    ax = ax.flatten()

    for i,img in enumerate(imlist):
        # correct color scale as images are loaded by openCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax[i].imshow(img)
        if i==0:
            ax[i].set_title('original')
        else:
            ax[i].set_title('lvl: '+str(i/10))
        # ax[i].axis('off')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

        for spine in ax[i].spines.values():
            spine.set_edgecolor('black')
    
    ax[11].axis('off')
    
    fig.suptitle('Noise levels for token: '+token)
    
    plt.tight_layout(pad=0)
    
    store_path = os.path.join(output_folder,'o3d_'+token+'.png')

    plt.savefig(store_path)
    plt.close()


    # matplotlib 3D reconstruction and saving in 3d plots:
    # Create the figure and axes object
    fig = plt.figure(figsize=(16, 9))

    for i,points in enumerate(pts_list):
        ax = fig.add_subplot(3,4,i+1, projection='3d')

        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        # Plot the 3D points
        ax.scatter(x, y, z, c='blue', marker='o')

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # set elevation angle view
        ax.view_init(elev=40, azim=180)

        # legend
        if i==0:
            ax.set_title('original')
        else:
            ax.set_title('lvl: '+str(i/10))

    # ax[11].axis('off')

    # Set title for the plot
    fig.suptitle('Noise levels for token: '+token)
    
    store_path = os.path.join(output_folder,'plt_'+token+'.png')
    
    plt.savefig(store_path) 
    plt.close()


    for i,img in enumerate(imlist):

        # Open3D reconstruction and saving in a subplot:
        fig, ax = plt.subplots(1,2,figsize=(16, 9))
        ax = ax.flatten()

        # correct color scale as images are loaded by openCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        if i==0:
            continue
        else:
            ax[0].imshow(imlist[0])
            ax[1].imshow(img)

            ax[0].set_title('original')
            ax[1].set_title('lvl: '+str(i/10))

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])

        for spine in ax[0].spines.values():
            spine.set_edgecolor('black')

        for spine in ax[1].spines.values():
            spine.set_edgecolor('black')
            
        fig.suptitle('Noise levels for token: '+token)
        
        plt.tight_layout(pad=0)
        
        store_path = os.path.join(output_folder,'o3d_comp_'+str(i)+'.png')

        plt.savefig(store_path)
        plt.close()



    for i,points in enumerate(pts_list):
        if i==0:
            continue

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(121, projection='3d')

        x = pts_list[0][:,0]
        y = pts_list[0][:,1]
        z = pts_list[0][:,2]

        # Plot the 3D points
        ax.scatter(x, y, z, c='blue', marker='o')

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # set elevation angle view
        ax.view_init(elev=40, azim=180)
        
        ax.set_title('original')



        ax = fig.add_subplot(122, projection='3d')

        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        # Plot the 3D points
        ax.scatter(x, y, z, c='blue', marker='o')

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # set elevation angle view
        ax.view_init(elev=40, azim=180)
        
        ax.set_title('lvl: '+str(i/10))



        # Set title for the plot
        fig.suptitle('Noise levels for token: '+token)
        
        store_path = os.path.join(output_folder,'plt_comp_'+str(i)+'.png')
        
        plt.savefig(store_path) 
        plt.close()


    exit()


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
                if 'CAM' in sensor:
                    continue

                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = 'nuScenes/'+sample_data['filename']

                print('sample_data:\n',sample_data)

                # get current ego vel in sensor frame
                deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)
                print('ego_vel:',deformer.ego_vel)

                # get_ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                # cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                # sensor_record = nusc.get('sensor', cs_record['sensor_token'])


                # print(get_ego_pose)
                # print(cs_record)
                # print(sensor_record)

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

                    if args.gen_lvl_grad_img:
                        noise_lvl_grad_gen(args,filename,sensor,deformer,radar_df)

                    if args.gen_csv:
                        sample_name = filename.split('/')[-1].split('.')[0]
                        radar_df.to_csv('./noisy_nuScenes/examples/RADAR/'+sample_name+'.csv')
                        continue
                    
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
                        save_radar_pointcloud(newfoldername,filename,pts_OG,pts_new,dat,newdat)

                        input('PRESS ENTER')


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
    # print('df to convert:')
    # print(df)
    print('converting df...')
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
                                                   'mid_range':40,
                                                   'far_range':4   # long range beam is +/- 9 degrees                                                    
                                                  },

                                            'resolution':{'near_range':0.39,
                                                          'far_range':1.79
                                                         },
                                            },
                                    'vel':{'range':[-111.11, 55.5],
                                            'resolution':{'near_range':0.119,
                                                          'far_range':0.103
                                                        }
                                            }
                                    }
        self.radar_dist_accuracy =  {'near_range' :0.1,
                                    'far_range': 0.4
                                    }
        self.radar_ang_accuracy =   {'near_range' :{'0':0.3,
                                                    '45':1,
                                                    '60':5
                                                    },
                                    'far_range': 0.1
                                    }
        self.radar_vel_accuracy = 0.1/3.6 


        # Ghost points appearing rate is independant of the noise level
        # => fixed, very small amount
        self.radar_ghost_max = 4
        self.rcs_max = 64 # from nuscenes


        self.args = args
        self.verbose = args.verbose

        # Radar noise level is defined as the 0.1 x the dB decrease between SNR' and SNR :
        # n = 0 : 0dB decrease
        # n = 0.1 : -1 dB decrease
        # n = 1 : -10 dB decrease

        self.noise_level_radar = args.n_level_radar
        self.SNR_ratio_dB = 10*log(10**(-self.noise_level_radar))
        self.SNR_ratio_linear = 10**(-self.noise_level_radar)

        self.noise_level_cam = args.n_level_cam

        self.ego_vel = np.array([0,0])
    
    #---------------------------------------------------------Radar functions---------------------------------------------------------
    def create_ghost_point(self, num_ghosts, radar_df, ghost_df):
        '''
        Generating fake points
        To better simulate the reception of points the generation is made in polar coordinates
        Velocity is generated in cartesian coordinate as we don't have enough information to generate it in polar coordinates.
        For more realistic velocities we sample from the current objects' and compensate from ego velocity reconstruction 

        Note on position generation: no need to check bounds as they are guaranteed by the uniform distribution
        '''
        print('initial ghost_df:',ghost_df)

        # setting max range for generated points
        # We want to avoid having easily-removable outliers
        max_range=max(radar_df['x'].to_numpy()) + 10 # range is withing sample distribution with a 10m additional margin
        if max_range+10>self.radar_sensor_bounds['dist']['range']['far_range']: 
            # max range cannot exceed radar actual bounds 
            max_range = self.radar_sensor_bounds['dist']['range']['far_range']
        
        print('max_range:',max_range)

        for i in range(num_ghosts):
            #---- Generating x,y,z coordinates ------

            print('ghost point n°',i)

            r = np.random.uniform(0.2,max_range)
            print('r:',r)

            if r<=self.radar_sensor_bounds['dist']['range']['short_range']:    # short range
                if self.verbose: print('point in short range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['short_range'],self.radar_sensor_bounds['dist']['ang']['short_range'])

            elif r<=self.radar_sensor_bounds['dist']['range']['mid_range']:    # mid range
                if self.verbose: print('point in mid range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['mid_range'],self.radar_sensor_bounds['dist']['ang']['mid_range'])

            else: # far range
                if self.verbose: print('point in far range')
                theta = np.random.uniform(-self.radar_sensor_bounds['dist']['ang']['far_range'],self.radar_sensor_bounds['dist']['ang']['far_range'])

            # converting back to cartesian coordinates
            x, y = polar_to_cart(r,theta)

            if self.verbose:
                print('ghost point',i)
                print('r_fake:',r)
                print('theta_fake:',theta)
                print('x_fake:',x)
                print('y_fake:',y)
                
            #---- Generating velocities ------
            '''
            Reminder: radars only measure a radial velocity, i.e. the projection of the relative velocity on the line of sight from the radar to the point.
            The vx and vy values are the components of the relative velocity, which is extracted by the radar by unknown methods
            vx_comp and vy_comp are the components of the motion-compensated radial velocity. The compensation process itself is unknown.
            '''
            r_vect = np.array((x,y))
            # r_mag = r # r = ||x,y||, r>0
            r_hat = r_vect/r

            # Because the process to acquire vx and vy is unknown, we can only sample a couple from the currrent distribtion
            # Note it's perfectly possible to sample an invalid point, which would be filtered afterwards.
            ID = np.random.choice(radar_df.index.to_list()) # this will also be useful for other parameters
            vx, vy = radar_df.loc[ID,['vx','vy']]
            v_vect = np.array([vx,vy])
            vr_vect = (np.dot(v_vect,r_hat))*r_hat # projection of V_vect on r_vect aka the radial velocity
            vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)            
            
            print('Velocities:')
            print('r_vect:',r_vect)
            print('r_hat:',r_hat)
            print('ID:',ID)
            print('corresponding row:\n',radar_df.loc[ID])
            print('v_vect:',v_vect)
            print('vr_vect:',vr_vect)
            print('vr_mag:',vr_mag)
            print('ego_vel:',self.ego_vel)

            if vr_mag==0:
                # point moves at our exact velocity or at radial projection blind spot
                vx_comp, vy_comp = (np.dot(self.ego_vel,r_hat))*r_hat # v_comp would be estimated ar our own vel
            else:
                vr_hat = vr_vect / vr_mag

                v_ego_r = (np.dot(self.ego_vel,r_hat))*r_hat # projecting ego velocity on radial vector
                v_ego_r_mag = sqrt(v_ego_r[0]**2+v_ego_r[1]**2)

                v_comp_mag = vr_mag - v_ego_r_mag
                vx_comp, vy_comp  = v_comp_mag * vr_hat
            
            
                print('vr_hat:',vr_hat)
                print('v_ego_r:',v_ego_r)
                print('v_ego_r_mag:',v_ego_r_mag)
                print('v_comp_mag:',v_comp_mag)
            print('v_comp:',[vx_comp,vy_comp])
            # input()            
            
            #---- Generating dynamic properties ------
            # dynProp: Dynamic property of cluster to indicate if is moving or not.
            # 0: moving
            # 1: stationary
            # 2: oncoming
            # 3: stationary candidate
            # 4: unknown
            # 5: crossing stationary
            # 6: crossing moving
            # 7: stopped
            # We retrieve dynamic property of the point based of the line we used to get v_x and v_y
            dyn_prop = radar_df.loc[ID,'dyn_prop']
            print('dyn_prop:',dyn_prop)
                       
            #---- RCS value ------
            # sorting the dataframe by rcs values and taking the lowest 25% values
            # then take uniform distribution amongst all
            rcs_dist = radar_df['rcs']
            print('rcs_dist = radar_df[\'rcs\']:',rcs_dist)
            rcs_dist.loc[len(rcs_dist)] = -5 #insuring -5 (smallest value) in the dataframe
            print('rcs_dist.loc[len(rcs_dist)] = -5:',rcs_dist)
            rcs_dist = rcs_dist.sort_values()
            print('sorted rcs_dist:',rcs_dist)
            rcs_dist = rcs_dist.drop_duplicates().reset_index(drop=True) # drop duplicate rcs
            print('drop_duplicates():',rcs_dist)
            rv = abs(np.random.normal(0,1/3)) # half gaussian
            print('rv:',rv)
            while rv>1:
                print('rv not good')
                rv = np.random.normal(0,1/3) # bounded upper norm
                print('new rv',rv)
            rcs_row = int(rv * len(rcs_dist)) # mapping bounded half-gaussian drawn random number to row in current distribution of rcs
            print('row:',rcs_row)

            rcs = rcs_dist.iloc[rcs_row]
            print('rcs_fake:',rcs)

            #---- misc properties ------
            # Taking most common values
            x_rms = radar_df['x_rms'].mode()[0]
            y_rms = radar_df['y_rms'].mode()[0]
            vx_rms = radar_df['vx_rms'].mode()[0]
            vy_rms = radar_df['vy_rms'].mode()[0]

            # valid states:
            # 0x00	valid                                                   (impossible --> ghost point)
            # 0x04	valid cluster with low RCS
            # 0x08	valid cluster with azimuth correction due to elevation  (impossible --> no elevation)
            # 0x09	valid cluster with high child probability
            # 0x0a	valid cluster with high probability of being a 50 deg artefact
            # 0x0b	valid cluster but no local maximum
            # 0x0c	valid cluster with high artefact probability
            # 0x0f	valid cluster with above 95m in near range              (impossible --> construction)
            # 0x10	valid cluster with high multi-target probability        (impossible --> ghost point)
            # 0x11	valid cluster with suspicious angle                     (impossible --> construction)
            invalid_state=np.random.choice([0x04,0x09,0x0a,0x0b,0x0c])
            print('x_rms:',x_rms)
            print('y_rms:',y_rms)
            print('vx_rms:',vx_rms)
            print('vy_rms:',vy_rms)
            print('invalid_state:',invalid_state)

            # pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
            # 0: invalid
            # 1: <25%
            # 2: 50%
            # 3: 75%
            # 4: 90%
            # 5: 99%
            # 6: 99.9%
            # 7: <=100%
            pdh0_val = 100 * abs(np.random.normal(0.5, 0.5/3)) #absolute value adds a little more chances of the 0+ side
            print('pdh0_val:',pdh0_val)
            while pdh0_val>100:
                print('pdh0_val not good')
                # removing potential cases > 100%
                pdh0_val = 100 * abs(np.random.normal(1, 1+1/3))
                print('new pdh0_val:',pdh0_val)

            bins = [25, 50, 75, 90, 99, 99.9, 100]  # Upper bounds for categories
            pdh0 = np.digitize(pdh0_val, bins) + 1  # Map to category (1 to 7)
            print('pdh0:',pdh0)
            # x  y  z  dyn_prop  id  rcs  vx  vy  vx_comp  vy_comp  is_quality_valid  ambig_state  x_rms  y_rms  invalid_state  pdh0  vx_rms  vy_rms
            row = [x, y, 0.0, dyn_prop, ID, rcs, vx, vy, vx_comp, vy_comp, 1, 3, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms]
            
            ghost_df.loc[i]=row

            print('final row:',ghost_df.loc[i])
            # input()

        return ghost_df

    def FP_FN_gen(self, radar_df):
        # Simulating ghost points and missed points
        print('original_df:\n',radar_df)
        print(50*'-','FP_FN',50*'-')
        # Initializing new dataframes and variables
        subset_df = copy.deepcopy(radar_df)
        n_rows=len(radar_df)

        #-------------------------------------Ghost points generation-------------------------------------
        print(50*'-','Ghost points',50*'-')
        # Initializing output df
        ghost_df = pd.DataFrame(columns=radar_df.columns)

        # Randomly draws how many ghost points will appear in this sample from a uniform distribution U(0,ghost_rate+1)
        num_ghosts = np.random.randint(low=0,high=self.radar_ghost_max+1)

        if self.verbose:
            print('Generating %d ghost point'%(num_ghosts))
            # input()

        if num_ghosts:
            # Generating random ghost points
            ghost_df = self.create_ghost_point(num_ghosts, radar_df, ghost_df)

            if self.verbose:
                print('ghost points:')
                print(ghost_df)
                # input()


        #----------------------------------------Random points drop----------------------------------------
        print(50*'-','RCS-based random drop',50*'-')
        # From the radar equation: SNR = k.(RCS/r**4/P_noise)

        x = radar_df['x'].to_numpy()
        y = radar_df['y'].to_numpy()
        rcs = radar_df['rcs'].to_numpy()

        # rcs_linear = 10**(rcs_range/10) # converting RCS to linear scale
        # rcs_new_linear = rcs_linear*(1-self.noise_level_radar) # adding noise

        r = np.array([sqrt(a**2+b**2) for a,b in zip(x,y)])
        
        alpha = np.array([10**(x/10) for x in rcs])/(r**4) # rcs need to be converted to linear scale (in m²)
        # alpha = rcs_linear/(r**4) 
        alpha_min = min(alpha)
        
        # SNR' is proportional to: (SNR_ratio x rcs/r^4)
        beta = (alpha*self.SNR_ratio_linear) + np.random.normal(0,min(alpha),n_rows)    # adding small fluctuation to represent physicall phenom.
        
        drop_indices = np.where(beta < alpha_min)[0]
        n_drops = len(drop_indices)

        print('r:',r)
        print('alpha:',alpha)
        print('beta:',beta)
        print('alpha_min:',alpha_min)
        print('SNR_ratio_linear:',self.SNR_ratio_linear)
        print(beta<alpha_min)
        print('drop_indices:\n',drop_indices)
        print('n_drops:',n_drops,'/',n_rows) 
        # input()       
        
        # # Low rcs points have a higher chance of being dropped.
        # # Lowest rcs is 5.0 => range is [-5,+inf]
        # rcs_range = radar_df['rcs'].to_numpy()
        # print('rcs_range_og:',rcs_range)
        # min_rcs = -10 # allows normalization without having min value becoming 0 (which would always be dropped)
        # max_rcs = max(rcs_range)
        
        # # Simulating RCS drop due to noise level (same as in noise_generation())
        # rcs_linear = 10**(rcs_range/10) # converting RCS to linear scale
        # rcs_new_linear = rcs_linear*(1-self.noise_level_radar) # adding noise
        # rcs_range = np.array([10*log(x,10) for x in rcs_new_linear]) # convert back to dBsm
        
        # print('rcs_range:',rcs_range)
        # print('min_rcs:',min_rcs)
        # print('max_rcs:',max_rcs)  

        # # normalize rcs range between [0,1]
        # rcs_range_norm = (rcs_range - min_rcs)/(max_rcs - min_rcs + 1e-8)

        # # rcs_sorted = np.sort(rcs_range_norm)
        # # rcs_id_sorted = np.argsort(rcs_range_norm)

        # # generating drop probability from a half-gaussian rv
        # drop_prob = abs(np.random.normal(0,1/3,n_rows))  # 3sigma = noise_level

        # drop_indices = rcs_range[np.where(rcs_range_norm <= drop_prob)[0]]

        # print('rcs_range_norm:',rcs_range_norm)
        # # print('rcs_sorted:',rcs_sorted)
        # # print('rcs_id_sorted:',rcs_id_sorted)
        # print('drop_prob:',drop_prob)

        if self.verbose: 
            print('Removing %d rows out of %d'%(len(drop_indices),len(radar_df)))
            print('Removed rows:',drop_indices)
            # input()

        subset_df = subset_df.drop(drop_indices, axis=0)
        subset_df.reset_index(drop=True, inplace=True)

        if self.verbose: print('Subset:', subset_df)

        input('finished FP_FN')
        print()
        return subset_df, ghost_df

    def gaussian_noise_gen(self, subset_df): 
        '''
        Generating n random points from a gaussian random distribution
        noise_split is a percentage, the amount of points is this a subset of radar_df
        We apply the noise uniformly to all point. Noise is drawn from a gaussian rv.
        We consider the potential measurement error to increase as rcs value decreases. Therefore low rcs-valued points have a higher
        possible noise-induced shift.
        '''
        print('\n',50*'-','Noise generation',50*'-')
        # Initialization
        noisy_df = copy.deepcopy(subset_df)
        # print(subset_df)
        n_rows = len(subset_df)

        for i in range(n_rows):
            x = subset_df.loc[i,'x']
            y = subset_df.loc[i,'y']
            vx = subset_df.loc[i,'vx']
            vy = subset_df.loc[i,'vy']            
            vx_comp = subset_df.loc[i,'vx_comp']
            vy_comp = subset_df.loc[i,'vy_comp']

            rcs = subset_df.loc[i,'rcs']

            print('row:',i)
            print('initial parameters:\n',subset_df.loc[i,:])
            print()

            # position shift
            r, theta = cart_to_polar(x,y)

            print('r, theta:',r,theta)

            # extracting theoretical accuracies
            if r>self.radar_sensor_bounds['dist']['range']['mid_range']:
                # long range point (possible physical fluctuations)
                min_dist_acc = self.radar_dist_accuracy['far_range']
                min_ang_acc = self.radar_ang_accuracy['far_range']
            else:
                # near range point
                min_dist_acc = self.radar_dist_accuracy['near_range']

                if abs(theta)<0.5:
                    # point @ 0° (with some margin)
                    min_ang_acc = self.radar_ang_accuracy['near_range']['0']
                if abs(theta)<45:
                    # point @ 45°
                    min_ang_acc = self.radar_ang_accuracy['near_range']['45']
                else:
                    # point @ 60° and above (possible physical fluctuations)
                    min_ang_acc = self.radar_ang_accuracy['near_range']['60']
            min_vel_accuracy = self.radar_vel_accuracy # Reported velocity accuracy : 0.1 km/h at all ranges
            
            print('min_dist_acc:',min_dist_acc)
            print('min_ang_acc:',min_ang_acc)
            print('min_vel_accuracy:',min_vel_accuracy)

            # SNR decreasing simulation
            # Given the Radar Equation: SNR = k x (RCS/r**4)/P_noise, k constant
            # We model the SNR behaviour as SNR' = SNR*((10*n_lvl)+1) in linear scale
            # n_lvl goes from 0 to 1 representing a dB increase of 0 to 10.4.
            # The term ((10*n_lvl)+1) is defined as SNR_ratio, which is the ratio between SNR' and SNR
            SNR_ratio = self.SNR_ratio_linear
            
            # From the Cramer-Rao bound: the measurement accuracy is invertly proportional to sqrt(SNR)
            # From the radar equation we know: SNR = k x rcs, so acc is invertly proportional to sqrt(RCS)
            # The accuracy from the datasheet is considered to be the max accuracy => acc @ RCS = 64dBsm
            # In reality the points have a different RCS. We can calculate the ratio between RCS_max and RCS_real
            rcs_ratio = rcs/self.rcs_max 
            rcs_ratio_linear = 10**(rcs_ratio/10) # converting RCS ratio to linear scale
           
            # Corrected accuracy
            dist_sigma = min_dist_acc / sqrt(SNR_ratio*rcs_ratio_linear)
            ang_sigma = min_ang_acc / sqrt(SNR_ratio*rcs_ratio_linear)
            vel_sigma = min_vel_accuracy / sqrt(SNR_ratio*rcs_ratio_linear)
            
            print('rcs:',rcs)
            print('rcs_ratio_linear:',rcs_ratio_linear)
            print('rcs_ratio:',rcs_ratio)
            print('SNR_ratio:',SNR_ratio)
            print('SNR_ratio*rcs_ratio:',SNR_ratio*rcs_ratio)

            print('dist_sigma:',dist_sigma)
            print('ang_sigma:',ang_sigma)
            print('vel_sigma:',vel_sigma)


            # Creating normally distributed random noise on angle and distance
            dist_noise = np.random.normal(0,dist_sigma)
            ang_noise =  np.random.normal(0,ang_sigma)  # in degrees
            
            # Applying noise to original values
            r_noisy = r + dist_noise
            theta_noisy = degrees(theta) + ang_noise
            
            # converting back to cartesian coordinates
            x_noisy, y_noisy = polar_to_cart(r_noisy,theta_noisy)
            
            # Modifying original values
            noisy_df.loc[i,'x'] = x_noisy
            noisy_df.loc[i,'y'] = y_noisy

            print('(x,y):',x,y)
            print('(r,theta):',r,theta)
            print('dist_noise:',dist_noise)
            print('ang_noise:',ang_noise)
            print('r_noisy:',r_noisy)
            print('theta_noisy:',theta_noisy)
            print('x_noisy:',x_noisy)
            print('y_noisy:',y_noisy)

            # Uncorrelated noise generation on velocity (radar velocity measurement in independant from position)
            
            # creating the relative velocity vector
            v_vect = np.array((vx,vy)) 
            v_mag = sqrt(vx**2+vy**2) # v_mag is the norm of v_vect, alpha is the angle between vx and v_mag
            if v_mag!=0:
                v_hat = v_vect/v_mag # normalized velocity vector
            else:
                v_hat = v_vect  # null vector

            # Creating a line-of-sight (LOS) vector
            r_vect = np.array((x,y)) # vector of point-to-radar axis from the origin (the radar)
            # r_mag = r              # already r = ||x,y||
            r_hat = r_vect/r         # normalized r, r always > 0

            # The radar actually measures V_R (Radial velocity) which is a projection of v_vect on the LOS axis
            vr_vect = (np.dot(v_vect,r_hat))*r_hat
            vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)

            # Calculating projection angle between v and vr
            psi = atan2(np.cross(v_vect,vr_vect),np.dot(v_vect,vr_vect))

            print('velocity:')
            print('v_vect:',v_vect)
            print('v_mag:',v_mag)
            print('v_hat:',v_hat)
            print('r_vect:',r_vect)
            print('r_hat:',r_hat)
            print('vr_vect:',vr_vect)
            print('vr_mag:',vr_mag)
            print('psi:',psi)

            if cos(psi)==0 or v_mag==0:
                # in this case directly add noise on relative velocity magnitude
                # these are usually flagged as invalid points
                vel_noise = np.random.normal(0,vel_sigma,2)   # 2 independant normally distributed noises
                v_noisy_x, v_noisy_y = v_vect + vel_noise
                v_comp_noisy_x, v_comp_noisy_y = np.array([vx_comp,vy_comp]) + vel_noise
            else:
                # Simulate noise on the measurement of V_r 
                vel_noise = np.random.normal(0,vel_sigma)   # normally distributed noise
                vr_noisy_mag = abs(vr_mag + vel_noise)      # adding noise to radial velocity, making sure the magnitude stays >=0

                # Applying noise to V
                v_noisy_mag = vr_noisy_mag/cos(psi)
                v_noisy_x, v_noisy_y = v_noisy_mag * v_hat
            
                # converting into vx_comp and vy_comp
                # vx_comp and vy_comp are motion compensated radial velocities
                # We can apply the noisy directly to vx_comp and vy_comp, assuming linear relationship between vr and v_comp

                #This or get noise components in x and y
                v_comp_mag, v_alpha = cart_to_polar(vx_comp,vy_comp)
                v_comp_mag_noisy = v_comp_mag + vel_noise*cos(psi)
                v_comp_noisy_x, v_comp_noisy_y = polar_to_cart(v_comp_mag_noisy,degrees(v_alpha))
                print('vr_noisy_mag:',vr_noisy_mag)
                print('v_noisy_mag:',v_noisy_mag)
                print('v_comp_mag:',v_comp_mag)
                print('v_alpha:',v_alpha)
                print('v_comp_mag_noisy:',v_comp_mag_noisy)
            

            print('vel_noise:',vel_noise)   
            print('v_noisy_x:',v_noisy_x)
            print('v_noisy_y:',v_noisy_y)
            print('v_comp_noisy_x:',v_comp_noisy_x)
            print('v_comp_noisy_y:',v_comp_noisy_y)

            # Modifying original values
            noisy_df.loc[i,'vx'] = v_noisy_x
            noisy_df.loc[i,'vy'] = v_noisy_y
            noisy_df.loc[i,'vx_comp'] = v_comp_noisy_x
            noisy_df.loc[i,'vy_comp'] = v_comp_noisy_y

            # misc properties:
            # same
        return noisy_df


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

        disp_img_cv2(output_img, title='fog test', block=True)


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
            print('original lenght:',len(radar_df),'| new length:',len(trunc_df))
            print('ghost dataframe:\n',ghost_df)
            input()

        # Adding noise on remaining points (non-generated)
        noisy_df = self.gaussian_noise_gen(trunc_df)

        if self.verbose:
            compare_df = pd.DataFrame(columns=['x_OG','x_new','y_OG','y_new','vx_OG','vx_new','vy_OG','vy_new','vx_comp_OG','vx_comp_new','vy_comp_OG','vy_comp_new'])
            compare_df[['x_OG','y_OG','vx_OG','vy_OG','vx_comp_OG','vy_comp_OG']] = trunc_df [['x','y','vx','vy','vx_comp','vy_comp']]
        
            compare_df[['x_new','y_new','vx_new','vy_new','vx_comp_new','vy_comp_new']] = noisy_df [['x','y','vx','vy','vx_comp','vy_comp']]

            print('Comparing original (subset) | noisy dataset:\n',compare_df)
            input()

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
            input()

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
    parser.add_argument('--n_level_cam', '-ncam','-cnoise' , type=float, default=0.1, help='Noise level for cams')
    parser.add_argument('--n_level_radar', '-nrad','-rnoise' , type=float, default=0.1, help='Noise level for radars')

    # Output config
    parser.add_argument('--out_root', type=str, default='./noisy_nuScenes', help='Noisy output folder')

    # Display
    parser.add_argument('--disp_all_data', action='store_true', default=False, help='Display mosaic with camera and radar original info')
    parser.add_argument('--disp_radar', action='store_true', default=False, help='Display original Radar point cloud and new one')
    parser.add_argument('--save_radar', action='store_true', default=False, help='Save screenshot of original Radar point cloud and new one')
    parser.add_argument('--disp_img', action='store_true', default=False, help='Display original Camera image and new one')
    parser.add_argument('--disp_all_img', action='store_true', default=False, help='Display mosaic of camera views')
    parser.add_argument('--gen_lvl_grad_img', action='store_true', default=False, help='generate output files for multiple noise levels')
    parser.add_argument('--gen_csv', action='store_true', default=False, help='generate csv file out of df (debug)')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Verbosity on|off')

    # Other
    parser.add_argument('--debug', action='store_true', default=False, help='Debug argument')
    parser.add_argument('--checksum', action='store_true', default=False, help='checks encoding/decoding of files')






    return parser

def check_args(args):
    sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']  

    assert args.split in ['train','val','test','mini'], 'Wrong split type'

    if args.sensor:
        assert args.sensor in sensor_list, 'Unknown sensor selected'    
   
    assert os.path.exists(args.nusc_root), 'Data folder at %s not found'%(args.nusc_root)

    if not os.path.exists(args.out_root):
        mkdir_if_missing(args.out_root)
                          

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

    #TODO : generate_dataset() wrapper

    exit('end of script')



'''
Running command:
python dataset_handler.py -kf --sensor <SENSOR> -v


some reading :

https://github.com/nutonomy/nuscenes-devkit/blob/05d05b3c994fb3c17b6643016d9f622a001c7275/python-sdk/nuscenes/utils/data_classes.py#L315
https://forum.nuscenes.org/t/detail-about-radar-data/173/5
https://forum.nuscenes.org/t/radar-vx-vy-and-vx-comp-vy-comp/283/4
https://conti-engineering.com/wp-content/uploads/2020/02/ARS-408-21_EN_HS-1.pdf

# Promising paper for impact of rcs fluctuation on accuracy :
https://ieeexplore.ieee.org/abstract/document/55565
https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/610920#:~:text=The%20transmitted%20power%20has%20an,target%20%5B7%2C%208%5D.

https://github.com/Gil-Mor/iFish
https://github.com/noahzn/FoHIS

'''