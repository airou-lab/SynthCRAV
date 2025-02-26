from math import *
import numpy as np
from pyquaternion import Quaternion

# def rot_z(theta):
#     theta=radians(theta)
#     return np.array([[cos(theta), -sin(theta), 0],
#                      [sin(theta),  cos(theta), 0],
#                      [         0,           0, 1]])

# print(rot_z(80).T)



from nuscenes import NuScenes
from utils.utils import *
import pandas as pd
nusc = NuScenes(version='v1.0-mini', dataroot='./nuScenes', verbose=True)

for scene in nusc.scene:
        print('scene:\n',scene)
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('nusc_sample:\n',nusc_sample)

        while True:
            for sensor in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']:
            # for sensor in ['RADAR_FRONT']:
            # for sensor in ['RADAR_FRONT_LEFT']:
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = 'nuScenes/'+sample_data['filename']
                sample_name = sample_data['filename'].split('/')[-1].split('.')[0]
                
                ego_vel = get_ego_vel(nusc,nusc_sample,sensor)[:2]

                df = pd.read_csv('./noisy_nuScenes/examples/RADAR/'+sample_name+'.csv')

                for i in range(len(df)):
                        x,y,vx,vy,vx_comp,vy_comp = df.loc[i,['x','y','vx','vy','vx_comp','vy_comp']]

                        r_vect = np.array([x,y])
                        r_mag = sqrt(x**2+y**2)
                        r_hat = r_vect/r_mag

                        v_vect = np.array([vx,vy])
                        v_mag = sqrt(vx**2+vy**2)

                        v_comp_vect = np.array((vx_comp,vy_comp))
                        v_comp_mag = sqrt(vx_comp**2+vy_comp**2)
                        
                        vr_vect = (np.dot(v_vect,r_hat))*r_hat
                        vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)

                        v_ego = ego_vel
                        v_ego_mag = sqrt(v_ego[0]**2+v_ego[1]**2)    

                        if v_mag and v_comp_mag and vr_mag and v_ego_mag:
                            # v_hat= v_vect/v_mag
                            # vr_hat = vr_vect/vr_mag 
                            # v_comp_hat= v_comp_vect/v_comp_mag
                            # v_ego_hat = v_ego/v_ego_mag

                            v_ego_proj = np.dot(v_ego,r_hat)*r_hat
                            v_ego_proj_mag = sqrt(v_ego_proj[0]**2+v_ego_proj[1]**2)

                        print('r_vect:',r_vect)
                        print('v_vect:',v_vect)
                        print('v_comp_vect:',v_comp_vect)
                        print('vr_vect:',vr_vect)
                        print('v_ego:',v_ego)
                        print('v_ego_proj',v_ego_proj)

                        print('\nCalc Comp_vel = :',vr_vect + v_ego_proj)
                        print('\nDfComp_vel = :',v_comp_vect)

                        input()