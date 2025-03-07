#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

# sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
#                 'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']



def accumulate_results(sensor):
    # original results
    result_path=os.path.join('.','frontier_outputs','original','metrics_summary.json')

    with open (result_path,'r') as resfile:
        res = json.load(resfile)

    mAP_list=[res['mean_ap']]
    NDS_list=[res['nd_score']]


    for i in range(10,110,10):
        result_path=os.path.join('.','frontier_outputs',sensor+'_frontier_'+str(i),'metrics_summary.json')
        
        with open (result_path,'r') as resfile:
            res = json.load(resfile)

        mAP_list.append(res['mean_ap'])
        NDS_list.append(res['nd_score'])


    return mAP_list, NDS_list


def create_figs(mAP_list, NDS_list, sensor):
    outfolder=os.path.join('.','frontier_outputs','viz')

    # defining noise levels axis
    n_lvls = [0,10,20,30,40,50,60,70,80,90,100]

    ## mAP only
    plt.figure(figsize=(16, 9))

    plt.plot(n_lvls, mAP_list, linestyle='--', color='tab:blue', alpha=0.5)
    
    # n_lvls_smooth = np.linspace(0, 100, 100)  # 300 points for smoothness
    # spline = make_interp_spline(n_lvls, mAP_list, k=2)
    # mAP_list_smooth = spline(n_lvls_smooth)    
    # plt.plot(n_lvls_smooth, mAP_list_smooth, linestyle='--', color='tab:blue', alpha=0.5)
    mAP_list_smooth = gaussian_filter1d(mAP_list, sigma=1)
    plt.plot(n_lvls, mAP_list_smooth, linestyle='-', color='tab:blue', alpha=1)
    

    plt.title('Evolution of mean Average Precision with noise levels')
    plt.xlabel("Noise levels (%)")
    plt.ylabel("mAP")

    plt.legend(['mAP','smoothed_mAP'])
    plt.grid(True)
    plt.xticks(n_lvls)
    # plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    outfile=os.path.join(outfolder,sensor+'_mAP.png')
    plt.savefig(outfile)


    ## NDS only
    plt.figure(figsize=(16, 9))

    plt.plot(n_lvls, NDS_list, linestyle='--', color='tab:orange', alpha=0.5)
    
    # n_lvls_smooth = np.linspace(0, 100, 100)  # 300 points for smoothness
    # spline = make_interp_spline(n_lvls, NDS_list, k=2)
    # NDS_list_smooth = spline(n_lvls_smooth)    
    # plt.plot(n_lvls_smooth, NDS_list_smooth, linestyle='--', color='tab:blue', alpha=0.5)
    NDS_list_smooth = gaussian_filter1d(NDS_list, sigma=1)
    plt.plot(n_lvls, NDS_list_smooth, linestyle='-', color='tab:orange', alpha=1)
    

    plt.title('Evolution of mean Average Precision with noise levels')
    plt.xlabel("Noise levels (%)")
    plt.ylabel("NDS")

    plt.legend(['NDS','smoothed_NDS'])
    plt.grid(True)
    plt.xticks(n_lvls)
    # plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    outfile=os.path.join(outfolder,sensor+'_NDS.png')
    plt.savefig(outfile)



    ## NDS and mAP
    plt.figure(figsize=(16, 9))

    # plt.plot(n_lvls,mAP_list)
    # plt.plot(n_lvls,NDS_list)

    plt.plot(n_lvls, mAP_list, linestyle='--', color='tab:blue', alpha=0.5)
    plt.plot(n_lvls, mAP_list_smooth, linestyle='-', color='tab:blue', alpha=1)
    plt.plot(n_lvls, NDS_list, linestyle='--', color='tab:orange', alpha=0.5)
    plt.plot(n_lvls, NDS_list_smooth, linestyle='-', color='tab:orange', alpha=1)






    plt.title('Evolution of mean Average Precision and Nuscenes Detection Score with noise levels')
    plt.xlabel("Noise levels (%)")
    plt.ylabel("Accuracies")

    plt.legend(['mAP','NDS'])
    plt.grid(True)
    plt.xticks(n_lvls)

    outfile=os.path.join(outfolder,sensor+'_NDS_mAP.png')
    plt.savefig(outfile)
    plt.close()



if __name__ == '__main__':

    # mAP_list, NDS_list = accumulate_results('cam')
    # create_figs(mAP_list, NDS_list, 'cam')

    mAP_list, NDS_list = accumulate_results('radar')
    print(mAP_list)
    print(NDS_list)
    create_figs(mAP_list, NDS_list, 'radar')