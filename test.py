# #-----------------------------------------------
# # Author : Mathis Morales                       
# # Email  : mathis-morales@outlook.fr             
# # git    : https://github.com/MathisMM            
# #-----------------------------------------------

# import os
# import numpy as np
# import pandas as pd
# from math import *
# from pyquaternion import Quaternion

# pd.set_option('display.max_rows', None)

# # # half bell shape curve:
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm

# # # Generate normal distribution
# # x = np.linspace(-3, 3, 1000)
# # y = norm.pdf(x, 0, 1)  # Standard normal PDF

# # # Apply absolute value
# # x_abs = np.abs(x)
# # y_abs = norm.pdf(x_abs, 0, 1) + norm.pdf(-x_abs, 0, 1)  # Symmetrized density

# # # Plot
# # plt.plot(x, y, label="Normal Distribution (N(0,1))", linestyle="dashed")
# # plt.plot(x_abs, y_abs, label="Absolute Value of Normal", color='red')
# # plt.axvline(0, linestyle="dotted", color="black")  # Reference line
# # plt.legend()
# # plt.title("Absolute Value of a Normal Distribution")
# # plt.show()

# # for item in os.listdir('./noisy_nuScenes/samples/RADAR_FRONT/examples/'):
# #     df = pd.read_excel('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+item)
# #     n_rows = len(df)
    
# #     rcs_range = df['rcs'].to_numpy()
# #     min_rcs = -10 # allows normalization without having min value becoming 0 (which would always be dropped)
# #     max_rcs = max(rcs_range)
    
# #     # normalize rcs range between [0,1]
# #     rcs_range_norm = (rcs_range - min_rcs)/(max_rcs - min_rcs + 1e-8)

# #     rcs_sorted = np.sort(rcs_range_norm)
# #     rcs_id_sorted = np.argsort(rcs_range_norm)

# #     noise_lvl = 0.5
# #     drop_prob = abs(np.random.normal(0,noise_lvl/3,n_rows))  # higher drop prob for low rcs objects


# #     drop_indices = rcs_id_sorted[np.where(rcs_sorted <= drop_prob)[0]]
    

   
# #     print('df:\n',df)
# #     print('rcs_range:\n',rcs_range)
# #     print('rcs_range_norm:\n',rcs_range_norm)
# #     print('rcs_sorted:\n',rcs_sorted)
# #     print('rcs_id_sorted:\n',rcs_id_sorted)
# #     print('drop_prob:\n',drop_prob)
# #     print('drop_indices:\n',drop_indices)
    

# #     # df = df.sort_values('rcs')
# #     # # print(df.sort_values('rcs'))
# #     # # print(min(df['rcs'].to_list()))

# #     # rcs_range = df['rcs'].to_numpy()

# #     # rcs_range_norm = (rcs_range - min(rcs_range))/(max(rcs_range)- min(rcs_range))

# #     # random_vals = np.random.rand(len(df))
    
# #     # drop_rate = 0.5
# #     # adjusted_rate = drop_rate * (1-rcs_range_norm)

# #     # drop_indices = np.where(random_vals <= adjusted_rate)[0]
    
# #     # print('df:\n',df)
# #     # print('rcs_range:\n',rcs_range)
# #     # print('rcs_range_norm:\n',rcs_range_norm)
# #     # print('random_vals:\n',random_vals)
# #     # print('adjusted_rate:\n',adjusted_rate)
# #     # print('drop_indices:\n',drop_indices)

# #     input()

# # while(1):
# #     ghost_rate = 3
# #     print(np.random.randint(low=0,high=ghost_rate+1))
# #     print(ghost_rate * np.random.randint(low=0,high=ghost_rate+1))
# #     print(int(100 * ghost_rate * np.random.randint(low=0,high=ghost_rate+1)))
# #     input()


# # for item in os.listdir('./noisy_nuScenes/samples/RADAR_FRONT/examples/'):
# #     df = pd.read_excel('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+item)
# #     x,y,vx,vy = df.loc[0,['x','y','vx','vy']]

# #     x = 4
# #     y = 4
# #     vx = 9
# #     vy = 0

# #     print(x,y,vx,vy)
# #     d = np.array([x,y])
# #     v = np.array([vx,vy])
# #     r_hat = d/sqrt(d[0]**2+d[1]**2)
# #     v_hat = v/sqrt(vx**2+vy**2)
# #     v_r = np.dot(v,d)/sqrt(x**2+y**2)*r_hat # Radial velocity

# #     print(np.dot(v,d)/sqrt(x**2+y**2))
# #     print(sqrt(v_r[0]**2+v_r[0]**2))
# #     exit()


# #     vr_hat = v_r/sqrt(v_r[0]**2+v_r[1]**2)
# #     cos_theta = np.dot(vr_hat,v_hat)

# #     print('d:',d)
# #     print('v:',v)
# #     print('v_r:',v_r)
# #     print('angle between vr and v')
# #     print('cos_theta:',cos_theta)
# #     print(degrees(acos(cos_theta)))
# #     print(degrees(atan2(np.cross(v,v_r),np.dot(v,v_r))))

# #     print()
# #     print('angle between v and r:')
# #     print(np.dot(r_hat,v_hat))
# #     print(degrees(acos(np.dot(r_hat,v_hat))))

# #     print('angle between v_r and r:')
# #     vr_hat = v_r/sqrt(v_r[0]**2+v_r[1]**2)
# #     print(np.dot(r_hat,vr_hat))
# #     print(degrees(acos(np.dot(r_hat,vr_hat))))


# #     input()

# # 7.800000190734863 3.700000047683716 -9 0
# # d: [7.80000019 3.70000005]
# # v: [-9  0]
# # v_r: [-7.34684023 -3.48503956]
# # angle between vr and v
# # cos_theta: 0.9035018434337445
# # 25.377743604747405
# # 25.377743604747405
# # def polar_to_cart(r,theta):

# #     x = r * cos(radians(theta))
# #     y = r * sin(radians(theta))

# #     return x,y

# # def cart_to_polar(x,y):

# #     theta = atan2(y,x)
# #     r = sqrt(x**2 + y**2)

# #     return r, theta



#                 # estimated_vel = (np.array(next_pose) -np.array(current_pose))*2  #(2fps)
#                 # print('estimated_vel:',estimated_vel)
#                 # # estimated_vel = np.array([estimated_vel[1],estimated_vel[0]])


#                 # sample_name = sample_data['filename'].split('/')[-1].split('.')[0]
#                 # # print('sample_name:',sample_name)

#                 # # print()
#                 # cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
#                 # # print(cs_record)

#                 # qw,qx,qy,qz = cs_record['rotation']
#                 # quaternion = Quaternion(qw,qx,qy,qz)
#                 # rotation_matrix = quaternion.rotation_matrix
                
#                 # cs_yaw_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

#                 # print('sensor:',sensor)
#                 # print('sensor angle:',degrees(cs_yaw_angle))


#                 # qw,qx,qy,qz = ego_pose['rotation']
#                 # quaternion = Quaternion(qw,qx,qy,qz)
#                 # rotation_matrix = quaternion.rotation_matrix
                
#                 # ego_yaw_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

#                 # print('ego_yaw_angle:',degrees(ego_yaw_angle))

#                 # ego_vel = np.array([estimated_vel[0]*cos(ego_yaw_angle) + estimated_vel[1]*sin(ego_yaw_angle),estimated_vel[1]*cos(ego_yaw_angle) - estimated_vel[0]*sin(ego_yaw_angle)])
#                 # print('angled vel:',ego_vel)


# from nuscenes import NuScenes
# from utils import *
# nusc = NuScenes(version='v1.0-mini', dataroot='./nuScenes', verbose=True)

# for scene in nusc.scene:
#         print('scene:\n',scene)
#         nusc_sample = nusc.get('sample', scene['first_sample_token'])
#         print('nusc_sample:\n',nusc_sample)

#         while True:
#             # for sensor in ['RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']:
#             for sensor in ['RADAR_FRONT']:
#             # for sensor in ['RADAR_FRONT_LEFT']:
#                 sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
#                 filename = 'nuScenes/'+sample_data['filename']

#                 ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
#                 cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
#                 sensor_record = nusc.get('sensor', cs_record['sensor_token'])

#                 print('sample_data:\n',sample_data)            
#                 print('ego_pose:\n',ego_pose)            
#                 print('cs_record:\n',cs_record)            
#                 print('sensor_record:\n',sensor_record)            

#                 current_pose = ego_pose['translation']

#                 next_token = nusc_sample['next']
#                 next_pose = nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', nusc_sample['next'])['data'][sensor])['ego_pose_token'])['translation']

#                 print('current_pose:',current_pose)
#                 print('next_pose:',next_pose)

#                 global_vel = (np.array(next_pose) -np.array(current_pose))*2  #(2fps), velocity in global coordinates

#                 qw,qx,qy,qz = ego_pose['rotation']
#                 quaternion = Quaternion(qw,qx,qy,qz)
#                 rotation_matrix = quaternion.rotation_matrix
                
#                 ego_yaw_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # minus sign cause we are going from global to ego

#                 print('ego_yaw_angle:',degrees(ego_yaw_angle))  # angle between global reference and ego
                
#                 def z_rot_inv(theta):
#                     return np.array([[cos(theta), sin(theta)],
#                                       [-sin(theta),  cos(theta)]])
                                
#                 ego_vel = z_rot_inv(ego_yaw_angle) @ global_vel[:2]

#                 print('ego_vel:',ego_vel)

#                 qw,qx,qy,qz = cs_record['rotation']
#                 quaternion = Quaternion(qw,qx,qy,qz)
#                 rotation_matrix = quaternion.rotation_matrix
                
#                 sensor_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

#                 print('sensor:',sensor)
#                 print('sensor angle:',degrees(sensor_angle))

#                 vel_sensor_frame = z_rot_inv(sensor_angle) @ ego_vel

#                 print('vel_sensor_frame:',vel_sensor_frame)




#                 sample_name = sample_data['filename'].split('/')[-1].split('.')[0]
#                 df = pd.read_csv('./noisy_nuScenes/examples/RADAR/'+sample_name+'.csv')
#                 # print('df:\n',df)

#                 for i in range(len(df)):
#                         x,y,vx,vy,vx_comp,vy_comp = df.loc[i,['x','y','vx','vy','vx_comp','vy_comp']]
                        
#                         # theta = atan2(y,x)
#                         # assert cos(theta)!=0,'cos'

#                         r_vect = np.array([x,y])
#                         r_mag = sqrt(x**2+y**2)
#                         r_hat = r_vect/r_mag

#                         v_vect = np.array([vx,vy])
#                         v_mag = sqrt(vx**2+vy**2)

#                         v_comp_vect = np.array((vx_comp,vy_comp))
#                         v_comp_mag = sqrt(vx_comp**2+vy_comp**2)
                        
#                         vr_vect = (np.dot(v_vect,r_hat))*r_hat
#                         vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)

#                         v_ego = vel_sensor_frame
#                         v_ego_mag = sqrt(v_ego[0]**2+v_ego[1]**2)    
#                         # alpha = atan2(v_ego[1],v_ego[0])              

#                         if v_mag and v_comp_mag and vr_mag and v_ego_mag:
#                             v_hat= v_vect/v_mag
#                             vr_hat = vr_vect/vr_mag 
#                             v_comp_hat= v_comp_vect/v_comp_mag
#                             v_ego_hat = v_ego/v_ego_mag

#                             v_ego_proj = np.dot(v_ego,r_hat)*r_hat
#                             v_ego_proj_mag = sqrt(v_ego_proj[0]**2+v_ego_proj[1]**2)

#                         print('r_vect:',r_vect)
#                         print('v_vect:',v_vect)
#                         print('v_comp_vect:',v_comp_vect)
#                         print('vr_vect:',vr_vect)
#                         print('v_ego:',v_ego)
#                         print('v_ego_proj',v_ego_proj)

#                         # print()
#                         # print('abs(vr_vect)-v_ego_proj:',abs(vr_vect)-v_ego_proj)
#                         # print()
#                         # print('v_comp_mag:',v_comp_mag)
#                         # print('vr_mag-v_ego_proj_mag:',vr_mag-v_ego_proj_mag)


#                         input()





#             input()
#             if nusc_sample['next'] == "":
#                 #GOTO next scene
#                 print("no next data in scene %s"%(scene['name']))
#                 break
#             else:
#                 #GOTO next sample
#                 next_token = nusc_sample['next']
#                 nusc_sample = nusc.get('sample', next_token)



# exit()





# for item in os.listdir('./noisy_nuScenes/examples/RADAR/'):
#     df = pd.read_csv('./noisy_nuScenes/examples/RADAR/'+item)

#     token = item.split('.')[0]



#     for i in range(len(df)):
#         x,y,vx,vy,vx_comp,vy_comp = df.loc[i,['x','y','vx','vy','vx_comp','vy_comp']]
        
#         x=10
#         y=0
#         vrx=9.5
#         vry=1

#         v_ego_x = 9
#         v_ego_y = 1

#         r_vect = np.array([x,y])
#         r_mag = sqrt(x**2+y**2)
#         r_hat = r_vect/r_mag

#         theta = atan2(y,x)


#         vr = np.array([vrx,vry])
#         vr_mag = sqrt(vrx**2+vry**2) 
#         vr_hat = vr/vr_mag
#         v_ego = np.array([v_ego_x,v_ego_y])

#         # print('theta:',theta)
#         # print('vr:',vr)
#         # print('v_ego:',v_ego)
#         # print('r_vect:',r_vect)
#         # print('r_mag:',r_mag)
#         # print('r_hat:',r_hat)
#         # print('np.dot(v_ego,r_hat):',np.dot(v_ego,r_hat))

#         # print('v_ego:',v_ego)
#         # print('vr:',vr)
        
#         # v_comp_x = vr[0] - v_ego[0]
#         print('v_comp:',vr-(v_ego*vr_hat))

#         exit()



#         theta = atan2(y,x)
#         assert cos(theta)!=0,'cos'

#         r_vect = np.array([x,y])
#         r_mag = sqrt(x**2+y**2)
#         r_hat = r_vect/r_mag

#         v_vect = np.array([vx,vy])
#         v_mag = sqrt(vx**2+vy**2)

#         v_comp_vect = np.array((vx_comp,vy_comp))
#         v_comp_mag = sqrt(vx_comp**2+vy_comp**2)
        
#         vr_vect = (np.dot(v_vect,r_hat))*r_hat
#         vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)


#         if v_mag and v_comp_mag and vr_mag:
#             v_hat= v_vect/v_mag
#             vr_hat = vr_vect/vr_mag 
#             v_comp_hat= v_comp_vect/v_comp_mag
            
#             v_ego = Vr

#             # print(np.dot(vr_hat,r_hat))
#             # print(np.dot(v_comp_hat,r_hat))
#             # print(np.dot(vr_hat,v_comp_hat))

#             # print(vr_vect,'\t',r_vect,'\t',v_comp)
#             input()





#         # v = np.array([vx,vy])
#         # v_comp = np.array([vx_comp,vy_comp])

#         # v_comp_hat = v_comp/(sqrt(v_comp[0]**2+v_comp[1]**2))

#         # vr_vect = (np.dot(v,r_hat))*r_hat

#         # v_comp = np.array((vx_comp,vy_comp))

#         # v_ego = vr_vect + v_comp

#         # vr_vect_hat = vr_vect/(sqrt(vr_vect[0]**2+vr_vect[1]**2))
#         # v_compt_hat = v_comp/(sqrt(v_comp[0]**2+v_comp[1]**2))


#         # v_ego_mag = sqrt(v_ego[0]**2+v_ego[1]**2)

#         # print(v)
#         # print(vr_vect)
#         # print(v_comp)
#         # print(v_ego)

#         # # print(df.loc[i])
#         # # print(np.dot(r_hat,v_hat))
#         # # print(np.dot(r_hat,v_comp_hat))
#         # # print(np.dot(v_hat,v_comp_hat))
#         # # print(np.dot(v_hat,v_comp_hat)>0)

#         # input()







# exit()

# from dataset_handler import *
# # compare_df = pd.DataFrame(columns=['x_OG','x_new','y_OG','y_new','vx_OG','vx_new','vy_OG','vy_new','vx_comp_OG','vx_comp_new','vy_comp_OG','vy_comp_new'])
# compare_df = pd.DataFrame(columns=['x_OG','x_0.1','x_0.2','x_0.3','x_0.4','x_0.5','x_0.6','x_0.7','x_0.8','x_0.9','x_1'])

# for i,item in enumerate(os.listdir('noisy_nuScenes/samples/RADAR_FRONT/noise_lvl/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178')):
#     if item[-3:]=='pcd':
#         df = decode_pcd_file('noisy_nuScenes/samples/RADAR_FRONT/noise_lvl/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178/'+item)
#         n = int(item.split('_')[-1].split('.')[0])
#         if n==0:
#            compare_df['x_OG'] = df['x']
#         elif n<10:
#             compare_df['x_0.'+str(n)] = df['x']
#         elif n ==10:
#             compare_df['x_1'] = df['x']

# print(compare_df)



# from math import *
# import numpy as np
# from nuscenes import NuScenes
# from utils.utils import *
# import pandas as pd
# pd.set_option('display.max_rows', None)
# nusc = NuScenes(version='v1.0-mini', dataroot='./nuScenes', verbose=True)


# for scene in nusc.scene:
#         print('scene:\n',scene)
#         nusc_sample = nusc.get('sample', scene['first_sample_token'])
#         print('nusc_sample:\n',nusc_sample)

#         while True:
#             for sensor in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']:
#                 sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
#                 filename = 'nuScenes/'+sample_data['filename']
#                 sample_name = sample_data['filename'].split('/')[-1].split('.')[0]
                
#                 df = pd.read_csv('./noisy_nuScenes/examples/RADAR/'+sample_name+'.csv')
                
#                 rcs = df['rcs'].sort_values().to_numpy()
#                 rcs_lin = 10**(rcs/10)
#                 rcs_noise = np.array([10 * log (x*(1-0.1),10) for x in rcs_lin])
#                 rcs_noise_round = np.array([round(x * 2) / 2] for x in rcs_noise)


#                 rcs_comp = pd.DataFrame({'rcs': rcs, 'rcs_lin': rcs_lin, 'rcs_noise': rcs_noise, 'rcs_noise_round': rcs_noise_round})
                
                
#                 print(rcs_comp)
#                 exit()
import numpy as np

while True:
    print(np.random.randint(1,11)/10)
    input()
