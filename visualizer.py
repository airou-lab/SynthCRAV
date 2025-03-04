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
from nuscenes import NuScenesExplorer
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud

from utils.utils import *

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

# Visualization and rendering functions
# idea to explore : render_pointcloud_in_image and render it on the relevant cam (all except cam_back)

nusc = None
args = None

def init_var(nusc_inst,args_inst):
    global nusc, args
    nusc = nusc_inst
    args = args_inst

def viz_nusc(nusc,singleSensor='CAM_FRONT'):
    # parse nusc and display camera views
    for scene in nusc.scene:
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('scene:\n',scene)
        print('nusc_sample:\n',nusc_sample)
        while True:
            sample_data = nusc.get('sample_data', nusc_sample['data'][singleSensor])
            filename = 'nuScenes/'+sample_data['filename']
            
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

def viz_all_cam_img(nusc_sample, save=False):
    # Visualize mosaic of all Camera images of this sample   
    fig, ax = plt.subplots(2,3)
    ax = ax.flatten()
    
    for i, sensor in enumerate(cam_list):
        sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
        filename = 'nuScenes/'+sample_data['filename']

        img = plt.imread(os.path.join('nuScenes',sample_data['filename']))
        ax[i].imshow(img)

        ax[i].set_title(sensor)

    if not save: 
        plt.show(block=True)
    else:
        figname = filename.split('/')[-1]
        mkdir_if_missing('./noisy_nuScenes/examples/CAM/cam_img/')
        plt.savefig('./noisy_nuScenes/examples/CAM/cam_img/'+figname)
    
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

def disp_img_plt(imgs=[],rows=10,cols=10,title='',legends=[],show=True, store_path=''):
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

        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

        for spine in ax[i].spines.values():
            spine.set_edgecolor('black')

    if title != '':
        fig.suptitle(title)
    
    plt.tight_layout(pad=0)
    
    if show: 
        plt.show()

    if store_path!='':
        plt.savefig(store_path) 
        plt.close()

def disp_img_cv2(img,title='',block=True, store_path=''):
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

def save_radar_3D_render(args,filename,pts_OG,pts_new,dat,newdat):
    # Saving radar point cloud rendering in matplotlib and in open3d bird eyeview
    sensor = filename.split('/')[2]
    sensor_type = sensor.split('_')[0]
    timestamp = filename.split('/')[-1].split('.')[0].split('_')[-1]
    output_folder = os.path.join(args.out_root,'examples',sensor_type,sensor,'imgs')
    mkdir_if_missing(output_folder)
    
    # Saving in matplotlib with auto angling for 3d effect
    image_path = os.path.join(output_folder,timestamp+'_plt_OG.png')
    disp_radar_pts(pts_OG,title='original',display=False, store_path=image_path)
    
    image_path = os.path.join(output_folder,timestamp+'_plt_new.png')
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
    image_path = os.path.join(output_folder,timestamp+'_o3d_OG.png')
    vis.capture_screen_image(image_path)
    vis.destroy_window()
    
    vis.create_window() 
    vis.add_geometry(newdat)
    vis.poll_events()
    vis.update_renderer()
    image_path = os.path.join(output_folder,timestamp+'_o3d_new.png')
    vis.capture_screen_image(image_path)
    vis.destroy_window()

def radar_df_to_excel(radar_df,filename):
    name = filename.split('/')[-1].split('.')[0]+'.xlsx'
    radar_df.to_excel   ('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+name)
    print('saved to %s'%('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+name))
    input()


# Noise level gradient generation
def noise_lvl_grad_gen_radar(args,filename,sensor,deformer,radar_df):
    '''
    Generates a gradient of noise levels in matplotlib subplot
    output folder is in out_root/examples
    '''
    token = filename.split('/')[-1].split('.')[0]

    output_folder = os.path.join(args.out_root,'examples','RADAR',sensor,'noise_lvl',token)
    print('generating gradient of noise levels at',output_folder)
    mkdir_if_missing(output_folder)

    token = filename.split('/')[-1].split('.')[0]
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
        deformer.noise_level_radar = (noise_lvl)/10
        deformer.update_val()
        print('noise level at %f %%'%(deformer.noise_level_radar * 100))
        print('corresonding dB SNR decrease:',round(deformer.SNR_decrease_dB,3),'dB')
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

    print('gathering o3d plots')
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

    print('o3d multiplot')
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

    print('matplotlib multiplot')
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

    print('o3d compare')
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


    print('matplotlib compare')
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

    print('Finished plotting all noise levels')
    input('PRESS ANY KEY')

def noise_lvl_grad_gen_cam(deformer,og_img,filename):
    #--------------------display a plot of each type at all noise levels----------------------
    token = filename.split('/')[-1].split('.')[0]
    sensor = filename.split('__')[1]

    for deform_type in ['Blur','High_exposure','Low_exposure','Gaussian_noise']:     
        print('Deforming image with', deform_type,'deformer')

        # init image list and legend list
        img_list =[og_img]
        legends = ['original']

        store_root=os.path.join('noisy_nuScenes','examples', 'CAM',sensor,'noise_level',token)
        mkdir_if_missing(store_root)

        for i in range(1,11,1):
            deformer.noise_level_cam = i/10                                               # Convert i to noise level
            print('generating noise level at: %d%%'%(int(deformer.noise_level_cam*100)))

            # Apply and store transform at level i/100
            noisy_img = deformer.deform_image(og_img,deform_type)
            img_list.append(noisy_img)

            # Also save a OG vs current noise plot
            store_path=os.path.join(store_root, deform_type+'_'+str(i)+'.png')
            disp_img_plt(imgs=[og_img,noisy_img],rows=1,cols=2,title=deform_type+' at '+str(deformer.noise_level_cam)+'%%',\
                        legends=['Original',str(deformer.noise_level_cam * 100)+'% noise'],show=False,store_path=store_path)
            
            # Store level legend
            legends.append(str(deformer.noise_level_cam * 100)+'% noise')

        store_path=os.path.join(store_root, deform_type+'.png')
        disp_img_plt(imgs=img_list,rows=3,cols=4,title=deform_type+' levels tests',legends=legends,show=False,store_path=store_path)
    
    print('Finished plotting all noise levels')
    input('PRESS ANY KEY')


def gen_paper_img_cam(deformer,og_img,filename):
    token = filename.split('/')[-1].split('.')[0]

    for deform_type in ['Blur','High_exposure','Low_exposure','Gaussian_noise']:     
        print('Deforming image with', deform_type,'deformer')

        # init image list and legend list
        img_list =[og_img]
        legends = ['original']

        store_root=os.path.join('noisy_nuScenes','examples','paper_img')
        mkdir_if_missing(store_root)

        for i in range(1,11,1):
            if i not in[0,3,6,10]:
                continue

            deformer.noise_level_cam = i/10                                               # Convert i to noise level
            print('generating noise level at: %d%%'%(int(deformer.noise_level_cam*100)))

            # Apply and store transform at level i/100
            noisy_img = deformer.deform_image(og_img,deform_type)
            img_list.append(noisy_img)
            
            # Store level legend
            legends.append(str(deformer.noise_level_cam * 100)+'% noise')

        store_path=os.path.join(store_root, deform_type+'-'+token+'.png')
        disp_img_plt(imgs=img_list,rows=2,cols=2,title='',legends=legends,show=False,store_path=store_path)
    
    print('Finished plotting all noise levels')
    input('PRESS ANY KEY')

def gen_paper_img_radar(args,filename,sensor,deformer,radar_df):
    '''
    Generates paper images
    '''
    token = filename.split('/')[-1].split('.')[0]

    output_folder = os.path.join(args.out_root,'examples','paper_img')
    print('generating gradient of noise levels at',output_folder)
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

    for noise_lvl in range(1,11,1):
        df_og = copy.deepcopy(radar_df)
        deformer.noise_level_radar = (noise_lvl)/10
        deformer.update_val()
        print('noise level at %f %%'%(deformer.noise_level_radar * 100))
        print('corresonding dB SNR decrease:',round(deformer.SNR_decrease_dB,3),'dB')
        newfilename = os.path.join(output_folder,token+'_'+str(noise_lvl)+'.pcd')
        
        deformed_radar_df = deformer.deform_radar(df_og)
        
        encode_to_pcd_file(deformed_radar_df,filename,newfilename)

        # Read datapoints
        newdat = o3d.io.read_point_cloud(newfilename)
        os.remove(newfilename)

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

    print('gathering o3d plots')
    # Gather o3d images in a list
    imlist=[]
    
    for i in range(11):
        name = 'o3d_'+str(i)+'.png'
        imlist.append(cv2.imread(os.path.join(output_folder,name)))

    print('o3d multiplot')
    # Open3D reconstruction and saving in a subplot:
    fig, ax = plt.subplots(2,2,figsize=(16, 9))
    ax = ax.flatten()

    k=0
    for i,img in enumerate(imlist):
        if i not in[0,3,6,10]:
            continue

        # correct color scale as images are loaded by openCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax[k].imshow(img)
        ax[k].set_title('%0.1f%% noise'%(i*10))

        # ax[i].axis('off')
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].set_xticklabels([])
        ax[k].set_yticklabels([])

        for spine in ax[k].spines.values():
            spine.set_edgecolor('black')
        k+=1
            
    plt.tight_layout(pad=1)
    
    store_path = os.path.join(output_folder,'o3d_'+token+'.png')

    plt.savefig(store_path)
    plt.close()

    print('matplotlib multiplot')
    # matplotlib 3D reconstruction and saving in 3d plots:
    # Create the figure and axes object
    fig = plt.figure(figsize=(16, 9))

    k=0
    for i,points in enumerate(pts_list):
        if i not in[0,3,6,10]:
            continue
        k+=1
        ax = fig.add_subplot(2,2,k, projection='3d')

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

        ax.set_title('%0.1f%% noise'%(i*10))
    
    store_path = os.path.join(output_folder,'plt_'+token+'.png')
    
    plt.savefig(store_path) 
    plt.close()

    
    print('Finished plotting paper img')
    input('PRESS ANY KEY')

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


