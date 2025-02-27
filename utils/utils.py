#-----------------------------------------------
# Author      : Mathis Morales                       
# Email       : mathis-morales@outlook.fr             
# git         : https://github.com/MathisMM            
#-----------------------------------------------

import os
# import sys 
import numpy as np
import pandas as pd
from typing import Tuple
from pyquaternion import Quaternion
from math import cos, sin, asin, acos, atan2, sqrt, radians, degrees, pi
import struct

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix




# Misc
def mkdir_if_missing(path):
    normpath = os.path.normpath(path)
    folder_list = normpath.split(os.sep)
    top_dir=folder_list.pop(0)

    for item in folder_list:
        top_dir = os.path.join(top_dir,item)

        if not os.path.isdir(top_dir):
            os.mkdir(top_dir)
            print("created directory at:",top_dir)

def get_data_info(data_root,split):
    if split=='mini':
        return '_mini'
    if split in ['train','val']:
        if 'mini' in data_root:
            return '_mini'
        else:
            return ''
    elif 'test' in split:
        return '_test'

# Math
def rot_x(theta):
    theta=radians(theta)
    return np.array([[1,          0,           0],
                     [0, cos(theta), -sin(theta)],
                     [0, sin(theta),  cos(theta)]])

def rot_y(theta):
    theta=radians(theta)
    return np.array([[ cos(theta), 0,  sin(theta)],
                     [          0, 1,           0],
                     [-sin(theta), 0,  cos(theta)]])

def rot_z(theta):
    theta=radians(theta)
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta),  cos(theta), 0],
                     [         0,           0, 1]])

def polar_to_cart(r,theta):

    x = r * cos(radians(theta))
    y = r * sin(radians(theta))

    return x,y

def cart_to_polar(x,y):

    theta = atan2(y,x)
    r = sqrt(x**2 + y**2)

    return r, theta

# Radar data encoder/decoder
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





# nuScenes functions utils
def load_nusc(split,data_root):
    assert split in ['train','val','test','mini'], "Bad nuScenes version"

    if split=='mini':
        nusc_version = 'v1.0-mini'
    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

def get_sensor_param(nusc, sample_token, cam_name='CAM_FRONT'):	# Unused function


    sample = nusc.get('sample', sample_token)

    # get camera sensor
    cam_token = sample['data'][cam_name]
    sd_record_cam = nusc.get('sample_data', cam_token)
    cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record_cam['ego_pose_token'])

    return pose_record, cs_record_cam

def get_sample_info(nusc,sensor,token,verbose=False):	# Unused function

    scenes = nusc.scene
    # print(scenes)
    # input()
    for scene in scenes:

        first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0

        while True:
            if sample_data['sample_token']==token:
                if verbose :
                    print('\nscene: ',scene)
                    print('\nsample: ',first_sample)
                    print ('\nsample_data: ',sample_data)
                return scene['name'], sample_data['filename']

            if sample_data['next'] == "":
                #GOTO next scene
                # print("no next data")
                if verbose:
                    print ('token NOT in:',scene['name'])
                break
            else:
                #GOTO next sample
                next_token = sample_data['next']
                sample_data = nusc.get('sample_data', next_token)

        # #Looping scene samples
        # while(sample_data['next'] != ""):       
        #     # if sample_token corresponds to token
        #     if sample_data['sample_token']==token:

        #         if verbose :
        #             print('\nscene: ',scene)
        #             print('\nsample: ',first_sample)
        #             print ('\nsample_data: ',sample_data)
        #         return scene['name'], sample_data['filename']

        #     else:
        #         # going to next sample
        #         sample_data = nusc.get('sample_data', sample_data['next'])

    return 0

def get_total_scenes_list(nusc,sensor): # Unused function
    scenes = nusc.scene
    scenes_list=[]
    for scene in scenes :
        scenes_list.append(scene['name'])
        # print (scene)
        # first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        # sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0
        # print(sample_data)
        # input()

    return scenes_list

def get_scenes_list(path):
    scenes_list = []

    # listing scenes
    for scene in os.listdir(path):
        scenes_list.append(scene) if scene.split('.')[-1]=='txt' else ''
    
    return scenes_list

def get_sample_metadata (nusc,sensor,token,verbose=False):
    scenes = nusc.scene
    
    if verbose:
        print('Looking for metadata for token: %s'%(token))

    for scene in scenes:

        first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0
        
        #Looping scene samples
        while(True):
            # if sample_token corresponds to token
            if sample_data['token']==token:
                if verbose:
                    print('\nscene:',scene)
                    print('\nfirst sample:',first_sample)
                    print('\nsample_data:',sample_data)

                    print('\nego token:',sample_data['ego_pose_token'])
                    print('\nsensor token:',sample_data['calibrated_sensor_token'],'\n')

                sd_record = nusc.get('sample_data', sample_data['token'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])

                if verbose:
                    print('-----------------------------------------------------')
                    print('sd_record: ',sd_record)
                    print('\n cs_record: ',cs_record)
                    print('\n sensor_record: ',sensor_record)
                    print('\n pose_record: ',pose_record)
                    print('\n cam_intrinsic: ',cam_intrinsic)
                    print('\n imsize: ',imsize)
                    print('-----------------------------------------------------')
                    print ('\n\n')
                    print ()

                return cs_record, pose_record, cam_intrinsic

            if sample_data['next']=="":
                # going to next scene
                break
            else:
                # going to next sample
                sample_data = nusc.get('sample_data', sample_data['next'])

        if verbose:
            print ('token NOT in:',scene['name'])
    return 0

def get_ego_pose(nusc,sensor,token,verbose=False):# Unused function
    _, pose_record, _ = get_sample_metadata(nusc,sensor,token)
    return pose_record

def get_sensor_data(nusc,sensor,token,verbose=False):# Unused function
    cs_record, _, cam_intrinsic = get_sample_metadata(nusc,sensor,token)
    return cs_record, cam_intrinsic

def render_box(self, im: np.ndarray, text: str, vshift:int = 0, hshift:int = 0,\
				view: np.ndarray = np.eye(3), normalize: bool = False, bottom_disp: bool = False,\
				colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth: int = 2 , text_scale : float = 0.5) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param text : str. Add any text to the image, by default the first letter is centered on the bbox 3d center.
    :param vshift/hshift : int. Add a vertical/horizontal shift to the bbox text.
    :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param bottom_disp: Display text under bounding box.
    :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
    :param linewidth: Linewidth for plot.
    :param text_scale: size of text.
    """
    corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                 (int(corners.T[i][0]), int(corners.T[i][1])),
                 (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    h = corners.T[3][1] - corners.T[0][1]
    l = corners.T[0][0] - corners.T[1][0]

    center = [center_bottom[0],center_bottom[1]-h/2]

    if bottom_disp:
        # adding a vertical shif of 0.8*h downwards
        vshift = int(0.8*h)

    # centering text
    hshift = -len(text)*10
    hshift = int(-abs(l)/2)

    if "pedestrian" in text: # fixing too low txt for pedestrian (different bbox type)
        vshift = int(0.6*h)

    cv2.putText(im,
                text,
                org=(int(center[0])+hshift, int(center[1])+vshift),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=text_scale,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_AA
                )

def get_ego_vel(nusc,nusc_sample,sensor):
    sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
    
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    # print(ego_pose)
    # print(cs_record)

    current_pose = ego_pose['translation']
    next_pose = nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', nusc_sample['next'])['data'][sensor])['ego_pose_token'])['translation']

    # ego vel in global coord
    ego_vel_global_1 = (np.array(next_pose) -np.array(current_pose))*2  #(2fps), velocity in global coordinates
    ego_vel_global = np.array([sqrt(ego_vel_global_1[0]**2+ego_vel_global_1[1]**2),0,0])

    # print(ego_vel_global)

    # # angle from global to ego
    # quaternion1 = Quaternion(ego_pose['rotation'])
    # rotation_matrix1 = quaternion1.rotation_matrix   
    # ego_yaw_angle = degrees(np.arctan2(rotation_matrix1[1, 0], rotation_matrix1[0, 0])) # from tracking pipeline
   
    # # ego vel in vel frame
    # ego_vel_ego = rot_z(ego_yaw_angle).T @ ego_vel_global

    # angle from ego to sensor
    quaternion2 = Quaternion(cs_record['rotation'])
    rotation_matrix2 = quaternion2.rotation_matrix 
    sensor_yaw_angle = degrees(np.arctan2(rotation_matrix2[1, 0], rotation_matrix2[0, 0]))

    # print(sensor)
    # print(sensor_yaw_angle)
    # ego vel in sensor frame
    ego_vel_sensor = rot_z(sensor_yaw_angle).T @ ego_vel_global
    return ego_vel_sensor


# colors and boxes
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def fixed_colors():
    f = open('color_scheme.txt','r')     # reading file
    text = f.readlines()
    color_name = []
    color_val_txt = []
    color_val = []
    for item in text:
        color_name.append(item.split(',')[0])
        color_val_txt.append(item.split('(')[1].split(')')[0])
    for color in color_val_txt:
        color_tmp = (float(color.split(',')[0])/255,float(color.split(',')[1])/255,float(color.split(',')[2])/255,)
        color_val.append(color_tmp)

    f.close()
    return color_val

def box_name2color(name):
    if name == 'car':
        c = (255,0,0)       # red
    
    elif name == 'pedestrian':
        c = (0,0,255)       # blue
    
    elif name == 'truck':
        c = (255,255,0)     # yellow
    
    elif name == 'bus':
        c = (255,0,255)     # magenta
    
    elif name == 'bicycle':
        c = (0,255,0)       # green
    
    elif name == 'motorcycle':
        c = (192,192,192)   # silver
    
    elif name == 'trailer':
        c = (165,42,42)     # brown

    else :
        c = (255,255,255)     # brown

    return c

