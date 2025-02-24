#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os
import numpy as np
import pandas as pd
from math import *

pd.set_option('display.max_rows', None)

# # half bell shape curve:
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Generate normal distribution
# x = np.linspace(-3, 3, 1000)
# y = norm.pdf(x, 0, 1)  # Standard normal PDF

# # Apply absolute value
# x_abs = np.abs(x)
# y_abs = norm.pdf(x_abs, 0, 1) + norm.pdf(-x_abs, 0, 1)  # Symmetrized density

# # Plot
# plt.plot(x, y, label="Normal Distribution (N(0,1))", linestyle="dashed")
# plt.plot(x_abs, y_abs, label="Absolute Value of Normal", color='red')
# plt.axvline(0, linestyle="dotted", color="black")  # Reference line
# plt.legend()
# plt.title("Absolute Value of a Normal Distribution")
# plt.show()

# for item in os.listdir('./noisy_nuScenes/samples/RADAR_FRONT/examples/'):
#     df = pd.read_excel('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+item)
#     n_rows = len(df)
    
#     rcs_range = df['rcs'].to_numpy()
#     min_rcs = -10 # allows normalization without having min value becoming 0 (which would always be dropped)
#     max_rcs = max(rcs_range)
    
#     # normalize rcs range between [0,1]
#     rcs_range_norm = (rcs_range - min_rcs)/(max_rcs - min_rcs + 1e-8)

#     rcs_sorted = np.sort(rcs_range_norm)
#     rcs_id_sorted = np.argsort(rcs_range_norm)

#     noise_lvl = 0.5
#     drop_prob = abs(np.random.normal(0,noise_lvl/3,n_rows))  # higher drop prob for low rcs objects


#     drop_indices = rcs_id_sorted[np.where(rcs_sorted <= drop_prob)[0]]
    

   
#     print('df:\n',df)
#     print('rcs_range:\n',rcs_range)
#     print('rcs_range_norm:\n',rcs_range_norm)
#     print('rcs_sorted:\n',rcs_sorted)
#     print('rcs_id_sorted:\n',rcs_id_sorted)
#     print('drop_prob:\n',drop_prob)
#     print('drop_indices:\n',drop_indices)
    

#     # df = df.sort_values('rcs')
#     # # print(df.sort_values('rcs'))
#     # # print(min(df['rcs'].to_list()))

#     # rcs_range = df['rcs'].to_numpy()

#     # rcs_range_norm = (rcs_range - min(rcs_range))/(max(rcs_range)- min(rcs_range))

#     # random_vals = np.random.rand(len(df))
    
#     # drop_rate = 0.5
#     # adjusted_rate = drop_rate * (1-rcs_range_norm)

#     # drop_indices = np.where(random_vals <= adjusted_rate)[0]
    
#     # print('df:\n',df)
#     # print('rcs_range:\n',rcs_range)
#     # print('rcs_range_norm:\n',rcs_range_norm)
#     # print('random_vals:\n',random_vals)
#     # print('adjusted_rate:\n',adjusted_rate)
#     # print('drop_indices:\n',drop_indices)

#     input()

# while(1):
#     ghost_rate = 3
#     print(np.random.randint(low=0,high=ghost_rate+1))
#     print(ghost_rate * np.random.randint(low=0,high=ghost_rate+1))
#     print(int(100 * ghost_rate * np.random.randint(low=0,high=ghost_rate+1)))
#     input()


# for item in os.listdir('./noisy_nuScenes/samples/RADAR_FRONT/examples/'):
#     df = pd.read_excel('./noisy_nuScenes/samples/RADAR_FRONT/examples/'+item)
#     x,y,vx,vy = df.loc[0,['x','y','vx','vy']]

#     x = 4
#     y = 4
#     vx = 9
#     vy = 0

#     print(x,y,vx,vy)
#     d = np.array([x,y])
#     v = np.array([vx,vy])
#     r_hat = d/sqrt(d[0]**2+d[1]**2)
#     v_hat = v/sqrt(vx**2+vy**2)
#     v_r = np.dot(v,d)/sqrt(x**2+y**2)*r_hat # Radial velocity

#     print(np.dot(v,d)/sqrt(x**2+y**2))
#     print(sqrt(v_r[0]**2+v_r[0]**2))
#     exit()


#     vr_hat = v_r/sqrt(v_r[0]**2+v_r[1]**2)
#     cos_theta = np.dot(vr_hat,v_hat)

#     print('d:',d)
#     print('v:',v)
#     print('v_r:',v_r)
#     print('angle between vr and v')
#     print('cos_theta:',cos_theta)
#     print(degrees(acos(cos_theta)))
#     print(degrees(atan2(np.cross(v,v_r),np.dot(v,v_r))))

#     print()
#     print('angle between v and r:')
#     print(np.dot(r_hat,v_hat))
#     print(degrees(acos(np.dot(r_hat,v_hat))))

#     print('angle between v_r and r:')
#     vr_hat = v_r/sqrt(v_r[0]**2+v_r[1]**2)
#     print(np.dot(r_hat,vr_hat))
#     print(degrees(acos(np.dot(r_hat,vr_hat))))


#     input()

# 7.800000190734863 3.700000047683716 -9 0
# d: [7.80000019 3.70000005]
# v: [-9  0]
# v_r: [-7.34684023 -3.48503956]
# angle between vr and v
# cos_theta: 0.9035018434337445
# 25.377743604747405
# 25.377743604747405
# def polar_to_cart(r,theta):

#     x = r * cos(radians(theta))
#     y = r * sin(radians(theta))

#     return x,y

# def cart_to_polar(x,y):

#     theta = atan2(y,x)
#     r = sqrt(x**2 + y**2)

#     return r, theta

for item in os.listdir('./noisy_nuScenes/examples/RADAR/'):
    df = pd.read_csv('./noisy_nuScenes/examples/RADAR/'+item)

    for i in range(len(df)):
        x,y,vx,vy,vx_comp,vy_comp = df.loc[i,['x','y','vx','vy','vx_comp','vy_comp']]
        
        x=10
        y=0
        vrx=-9.5
        vry=0

        v_ego_x = 9
        v_ego_y = 0

        r_vect = np.array([x,y])
        r_mag = sqrt(x**2+y**2)
        r_hat = r_vect/r_mag

        theta = atan2(y,x)


        vr = np.array([vrx,vry])
        v_ego = np.array([v_ego_x,v_ego_y])

        print('theta:',theta)
        print('vr:',vr)
        print('v_ego:',v_ego)
        print('r_vect:',r_vect)
        print('r_mag:',r_mag)
        print('r_hat:',r_hat)
        print('np.dot(v_ego,r_hat):',np.dot(v_ego,r_hat))

        print(vr-(v_ego*r_hat))

        exit()



        theta = atan2(y,x)
        assert cos(theta)!=0,'cos'

        r_vect = np.array([x,y])
        r_mag = sqrt(x**2+y**2)
        r_hat = r_vect/r_mag

        v_vect = np.array([vx,vy])
        v_mag = sqrt(vx**2+vy**2)

        v_comp_vect = np.array((vx_comp,vy_comp))
        v_comp_mag = sqrt(vx_comp**2+vy_comp**2)
        
        vr_vect = (np.dot(v_vect,r_hat))*r_hat
        vr_mag = sqrt(vr_vect[0]**2+vr_vect[1]**2)


        if v_mag and v_comp_mag and vr_mag:
            v_hat= v_vect/v_mag
            vr_hat = vr_vect/vr_mag 
            v_comp_hat= v_comp_vect/v_comp_mag
            
            v_ego = Vr

            # print(np.dot(vr_hat,r_hat))
            # print(np.dot(v_comp_hat,r_hat))
            # print(np.dot(vr_hat,v_comp_hat))

            # print(vr_vect,'\t',r_vect,'\t',v_comp)
            input()





        # v = np.array([vx,vy])
        # v_comp = np.array([vx_comp,vy_comp])

        # v_comp_hat = v_comp/(sqrt(v_comp[0]**2+v_comp[1]**2))

        # vr_vect = (np.dot(v,r_hat))*r_hat

        # v_comp = np.array((vx_comp,vy_comp))

        # v_ego = vr_vect + v_comp

        # vr_vect_hat = vr_vect/(sqrt(vr_vect[0]**2+vr_vect[1]**2))
        # v_compt_hat = v_comp/(sqrt(v_comp[0]**2+v_comp[1]**2))


        # v_ego_mag = sqrt(v_ego[0]**2+v_ego[1]**2)

        # print(v)
        # print(vr_vect)
        # print(v_comp)
        # print(v_ego)

        # # print(df.loc[i])
        # # print(np.dot(r_hat,v_hat))
        # # print(np.dot(r_hat,v_comp_hat))
        # # print(np.dot(v_hat,v_comp_hat))
        # # print(np.dot(v_hat,v_comp_hat)>0)

        # input()







exit()

from dataset_handler import *
# compare_df = pd.DataFrame(columns=['x_OG','x_new','y_OG','y_new','vx_OG','vx_new','vy_OG','vy_new','vx_comp_OG','vx_comp_new','vy_comp_OG','vy_comp_new'])
compare_df = pd.DataFrame(columns=['x_OG','x_0.1','x_0.2','x_0.3','x_0.4','x_0.5','x_0.6','x_0.7','x_0.8','x_0.9','x_1'])

for i,item in enumerate(os.listdir('noisy_nuScenes/samples/RADAR_FRONT/noise_lvl/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178')):
    if item[-3:]=='pcd':
        df = decode_pcd_file('noisy_nuScenes/samples/RADAR_FRONT/noise_lvl/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178/'+item)
        n = int(item.split('_')[-1].split('.')[0])
        if n==0:
           compare_df['x_OG'] = df['x']
        elif n<10:
            compare_df['x_0.'+str(n)] = df['x']
        elif n ==10:
            compare_df['x_1'] = df['x']

print(compare_df)
