#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import numpy as np
import pandas as pd
import open3d as o3d
import copy
import argparse
import cv2
from math import cos, sin, asin, acos, atan2, sqrt, radians, degrees, pi, log

# Some useful functions
from utils.utils import *
from utils.fisheye import generate_fisheye_dist
from visualizer import *
import visualizer

pd.set_option('display.max_rows', None)

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

cam_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

radar_list = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']


# Dataset Parser (using kf because the other data aren't annotated and are interpolated)
def parse_nusc_keyframes(nusc, sensors, args, deformer):
    for scene in nusc.scene:
        nusc_sample = nusc.get('sample', scene['first_sample_token'])
        print('\nscene:\n',scene)

        while True:
            # Extract sensor data
            for sensor in sensors :
                # Load nusc info
                sample_data = nusc.get('sample_data', nusc_sample['data'][sensor])
                filename = os.path.join(args.nusc_root,sample_data['filename'])            
                
                if not args.debug:
                    print(150*' ',end='\r',flush=True)
                    print('current file:',filename,end='\r',flush=True)

                # Setting up output folder
                newfoldername = os.path.join(args.out_root,'samples', sensor, str(int(deformer.noise_level_radar*100)))
                mkdir_if_missing(newfoldername)

                if args.verbose:
                    print('nusc_sample:\n',nusc_sample)
                    print('sample_data:\n',sample_data)
                    print('Output folder name:',newfoldername)


                ##RADAR DATA SYNTHESIZER##
                if 'RADAR' in sensor:
                    # set output pcd file name
                    newfilename = os.path.join(newfoldername,filename.split('/')[-1])

                    # get current ego vel in sensor frame
                    deformer.ego_vel= get_ego_vel(nusc,nusc_sample,sensor)[:2] # only (vx,vy)
                    
                    if args.verbose: 
                        print('output filename:',newfilename)
                        print('ego_vel:',deformer.ego_vel)

                    radar_df,_ = decode_pcd_file(filename,args.verbose)
                    
                    if radar_df.isna().any().any():
                        # Empty original radar point cloud check
                        print(150*'',end='\r')# clear print
                        print('NaN value in dataframe: skipped')
                        encode_to_pcd_file(radar_df,filename,newfilename,args.verbose)  # copy pasting this cloud
                        continue

                    if args.gen_lvl_grad_img or args.gen_csv or args.gen_paper_img:
                        
                        if args.gen_lvl_grad_img:
                            noise_lvl_grad_gen_radar(args,filename,sensor,deformer,radar_df)

                        if args.gen_csv:
                            sample_name = filename.split('/')[-1].split('.')[0]
                            mkdir_if_missing('./noisy_nuScenes/examples/RADAR/csv/')
                            radar_df.to_csv('./noisy_nuScenes/examples/RADAR/csv/'+sample_name+'.csv')
                    
                        if args.gen_paper_img and sensor=='RADAR_FRONT':
                            gen_paper_img_radar(args, filename, sensor, deformer, radar_df)
                        continue

                    # Apply deformation
                    deformed_radar_df = deformer.deform_radar(radar_df)

                    if len(deformed_radar_df) ==0:
                        # Empty generated radar point cloud check
                        print(150*'',end='\r')# clear print
                        print('Empty dataframe generated')
                                                 # x    y        z   dyn_prop  id  rcs   vx   vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
                                                 # NaN  NaN      0.0        0  -1  0.0  0.0  0.0     0.0     0.0                0           0     0     0             0    0      0      0
                        deformed_radar_df.loc[0]=[np.nan, np.nan,0.0,       0, -1, 0.0, 0.0, 0.0,    0.0,    0.0,               0,          0,    0,    0,            0,   0,     0,     0]

                    # Save output
                    encode_to_pcd_file(deformed_radar_df,filename,newfilename,args.verbose)

                    if args.disp_radar or args.save_radar_render:
                        # Read datapoints
                        dat = o3d.io.read_point_cloud(filename)
                        newdat = o3d.io.read_point_cloud(newfilename)

                        # converting to numpy format
                        pts_OG = np.asarray(dat.points)
                        pts_new = np.asarray(newdat.points)

                        if args.disp_radar:
                            # TODO: clean up a bit
                            disp_radar_pts(pts_OG,title='original',display=True, store_path='')
                            disp_radar_pts(pts_new,title='new',display=True, store_path='')
                            
                            # # using open3d built in (no axis)
                            # print('Original point clound')
                            # # print(np.asarray(dat.points))
                            # # print(dat)
                            # o3d.visualization.draw_geometries([dat])

                            # print('New point clound')
                            # # print(np.asarray(newdat.points))
                            # # print(newdat)
                            # o3d.visualization.draw_geometries([newdat])

                            # dat = o3d.io.read_point_cloud(filename)
                            # viz_radar_dat(sample_data)
                            # o3d.visualization.draw_geometries([dat])

                        if args.save_radar_render:
                            save_radar_3D_render(args,filename,pts_OG,pts_new,dat,newdat)


                ##CAMERA DATA SYNTHESIZER##
                elif 'CAM' in sensor:
                    if args.gen_lvl_grad_img or args.gen_paper_img:
                        # Loading image from filename        
                        img = cv2.imread(filename)

                        if args.gen_lvl_grad_img:
                            noise_lvl_grad_gen_cam(deformer,img,filename)
                        
                        if args.gen_paper_img and  sensor == 'CAM_FRONT':
                            gen_paper_img_cam(args, filename, sensor, deformer, radar_df)
                        continue

                    for deform_type in ['Blur','High_exposure','Low_exposure','Gaussian_noise']:
                        if 'night' in scene['description'].lower():
                            # not considering data that has already low exposure and gaussian noise due to nighttime
                            continue

                        # Loading image from filename        
                        img = cv2.imread(filename)

                        # Output directory and file name
                        mkdir_if_missing(os.path.join(newfoldername,deform_type))
                        newfilename = os.path.join(newfoldername,deform_type,filename.split('/')[-1])
                        if args.verbose: print('output filename:',newfilename)

                        # Apply deformation
                        deformed_img = deformer.deform_image(img,deform_type)

                        # Save output
                        cv2.imwrite(newfilename, deformed_img)
                
                # next sensor
                if args.debug:
                    input('PRESS ANY KEY FOR NEXT SENSOR')

            if nusc_sample['next'] == "":
                #GOTO next scene
                print("no next data in scene %s"%(scene['name']))
                break
            else:
                #GOTO next sample
                next_token = nusc_sample['next']
                nusc_sample = nusc.get('sample', next_token)


# auto-generation wrapper
def genDataset(nusc, sensors, args):
    '''
    Generating all noise levels for both sensors
    For Radars the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<name.pcd>
    For Cameras the results are stored in noisy_nuScenes/samples/sensor/<noise_level>/<noise_type>/<name.jpg>
    Resulting data is 50 x bigger than original dataset    
    '''

    for noise_level in range(10,110,10):
        deformer=deform_data(args)
        deformer.noise_level_radar = noise_level/100
        deformer.noise_level_cam = noise_level/100
        deformer.update_val()

        print(50*'-','Generating data at %d %% noise'%(noise_level),50*'-')

        parse_nusc_keyframes(nusc, sensors, args, deformer)





# Deformer class
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
        self.SNR_decrease_dB = 10*log(10**(-self.noise_level_radar),10)
        self.SNR_ratio_linear = 10**(-self.noise_level_radar)

        self.noise_level_cam = args.n_level_cam

        self.ego_vel = np.array([0,0])
    
    def update_val(self):
        # updates values based on noise_level if it is tuned by other functions
        self.SNR_decrease_dB = 10*log(10**(-self.noise_level_radar),10)
        self.SNR_ratio_linear = 10**(-self.noise_level_radar)

    #---------------------------------------------------------Radar functions---------------------------------------------------------
    def create_ghost_point(self, num_ghosts, radar_df, ghost_df):
        '''
        Generating fake points
        To better simulate the reception of points the generation is made in polar coordinates
        Velocity is generated in cartesian coordinate as we don't have enough information to generate it in polar coordinates.
        For more realistic velocities we sample from the current objects' and compensate from ego velocity reconstruction 

        Note on position generation: no need to check bounds as they are guaranteed by the uniform distribution
        '''

        # setting max range for generated points
        # We want to avoid having easily-removable outliers
        max_range=max(radar_df['x'].to_numpy()) + 10 # range is withing sample distribution with a 10m additional margin
        if max_range+10>self.radar_sensor_bounds['dist']['range']['far_range']: 
            # max range cannot exceed radar actual bounds 
            max_range = self.radar_sensor_bounds['dist']['range']['far_range']

        
        for i in range(num_ghosts):
            #---- Generating x,y,z coordinates ------
            r = np.random.uniform(0.2,max_range)

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

            if self.verbose>=2:
                print('ghost point',i)
                print('max_range:',max_range)
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
            
            if self.verbose>=2:
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
            
                if self.verbose>=2:
                    print('vr_hat:',vr_hat)
                    print('v_ego_r:',v_ego_r)
                    print('v_ego_r_mag:',v_ego_r_mag)
                    print('v_comp_mag:',v_comp_mag)
            
            if self.verbose>=2: 
                print('v_comp:',[vx_comp,vy_comp])
            
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
            if self.verbose: print('dyn_prop:',dyn_prop)
                       
            #---- RCS value ------
            # sorting the dataframe by rcs values and taking the lowest 25% values
            # then take uniform distribution amongst all
            rcs_dist = radar_df['rcs']
            rcs_dist.loc[len(rcs_dist)] = -5 #insuring -5 (smallest value) in the dataframe
            rcs_dist = rcs_dist.sort_values()
            rcs_dist = rcs_dist.drop_duplicates().reset_index(drop=True) # drop duplicate rcs
            
            # Drawing a random rcs value (with high chance to draw a small one)
            rv = abs(np.random.normal(0,1/3)) # half gaussian            
            while rv>1:
                rv = np.random.normal(0,1/3) # bounded upper norm
            rcs_row = int(rv * len(rcs_dist)) # mapping bounded half-gaussian drawn random number to row in current distribution of rcs

            rcs = rcs_dist.iloc[rcs_row]
            
            if self.verbose>=2:
                print('point cloud rcs:',radar_df['rcs'])
                print('filtered rcs:',rcs_dist)
                print('drawn value:',rv)
                print('corresponding row:',rcs_row)
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
            
            if self.verbose>=2:
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
            while pdh0_val>100:
                # removing potential cases > 100%
                pdh0_val = 100 * abs(np.random.normal(1, 1+1/3))

            bins = [25, 50, 75, 90, 99, 99.9, 100]  # Upper bounds for categories
            pdh0 = np.digitize(pdh0_val, bins) + 1  # Map to category (1 to 7)
            if self.verbose: print('pdh0:',pdh0)
           
            # x  y  z  dyn_prop  id  rcs  vx  vy  vx_comp  vy_comp  is_quality_valid  ambig_state  x_rms  y_rms  invalid_state  pdh0  vx_rms  vy_rms
            row = [x, y, 0.0, dyn_prop, ID, rcs, vx, vy, vx_comp, vy_comp, 1, 3, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms]
            
            ghost_df.loc[i]=row

            if self.verbose>=2:
                print('final row:',ghost_df.loc[i])

        return ghost_df

    def FP_FN_gen(self, radar_df):
        # Simulating ghost points and missed points
        if self.verbose: 
            print(50*'-','FP_FN',50*'-')
        
        # Initializing new dataframes and variables
        subset_df = copy.deepcopy(radar_df)
        n_rows=len(radar_df)

        #-------------------------------------Ghost points generation-------------------------------------
        # Initializing output df
        ghost_df = pd.DataFrame(columns=radar_df.columns)

        # Randomly draws how many ghost points will appear in this sample from a uniform distribution U(0,ghost_rate+1)
        num_ghosts = np.random.randint(low=0,high=self.radar_ghost_max+1)

        if self.verbose:
            print(50*'-','Ghost points',50*'-')
            print('Generating %d ghost point'%(num_ghosts))

        if num_ghosts:
            # Generating random ghost points
            ghost_df = self.create_ghost_point(num_ghosts, radar_df, ghost_df)

            if self.verbose:
                print('ghost points:')
                print(ghost_df)


        #----------------------------------------Random points drop----------------------------------------
        if self.verbose: print(50*'-','RCS-based random drop',50*'-')
        # From the radar equation: SNR = k.(RCS/r**4/P_noise)

        x = radar_df['x'].to_numpy()
        y = radar_df['y'].to_numpy()
        rcs = radar_df['rcs'].to_numpy()

        r = np.array([sqrt(a**2+b**2) for a,b in zip(x,y)])
        
        alpha = np.array([10**(x/10) for x in rcs])/(r**4) # rcs need to be converted to linear scale (in m²)
        alpha_min = min(alpha)
        
        # SNR' is proportional to: (SNR_ratio x rcs/r^4)
        beta = (alpha*self.SNR_ratio_linear) + np.random.normal(0,min(alpha),n_rows)    # adding small fluctuation to represent physicall phenom.
        
        drop_indices = np.where(beta < alpha_min)[0]
        n_drops = len(drop_indices)
        
        subset_df = subset_df.drop(drop_indices, axis=0)
        subset_df.reset_index(drop=True, inplace=True)
        
        if self.verbose:
            print('r:',r)
            print('alpha:',alpha)
            print('beta:',beta)
            print('alpha_min:',alpha_min)
            print('SNR_ratio_linear:',self.SNR_ratio_linear)
            print(beta<alpha_min)
            print('drop_indices:\n',drop_indices)
            print('n_drops:',n_drops,'/',n_rows) 

            print('Removing %d rows out of %d'%(len(drop_indices),len(radar_df)))
            print('Removed rows:',drop_indices)

            print('Subset:', subset_df)
            input()

        return subset_df, ghost_df

    def gaussian_noise_gen(self, subset_df): 
        '''
        Generating n random points from a gaussian random distribution
        noise_split is a percentage, the amount of points is this a subset of radar_df
        We apply the noise uniformly to all point. Noise is drawn from a gaussian rv.
        We consider the potential measurement error to increase as rcs value decreases. Therefore low rcs-valued points have a higher
        possible noise-induced shift.
        '''
        if self.verbose: 
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

            # position shift
            r, theta = cart_to_polar(x,y)

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

            if self.verbose>=2:
                print('row:',i)
                print('initial parameters:\n',subset_df.loc[i,:])
                print('r, theta:',r,theta)
                print('min_dist_acc:',min_dist_acc)
                print('min_ang_acc:',min_ang_acc)
                print('min_vel_accuracy:',min_vel_accuracy)

                print('rcs:',rcs)
                print('rcs_ratio_linear:',rcs_ratio_linear)
                print('rcs_ratio:',rcs_ratio)
                print('SNR_ratio:',SNR_ratio)
                print('SNR_ratio*rcs_ratio:',SNR_ratio*rcs_ratio)

                print('dist_sigma:',dist_sigma)
                print('ang_sigma:',ang_sigma)
                print('vel_sigma:',vel_sigma)

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

            if self.verbose>=2:
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
                
                if self.verbose>=2:
                    print('vr_noisy_mag:',vr_noisy_mag)
                    print('v_noisy_mag:',v_noisy_mag)
                    print('v_comp_mag:',v_comp_mag)
                    print('v_alpha:',v_alpha)
                    print('v_comp_mag_noisy:',v_comp_mag_noisy)
            
            if self.verbose>=2:
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
            print('current noise lvl:', self.noise_level_radar)
            print('corresponding dB decrease:',self.SNR_decrease_dB)
            print('corresponding linear ratio:',self.SNR_ratio_linear)
            input()

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
      
    def deform_image(self,img,deform_type):

        if deform_type == 'Blur':
            if args.verbose:
                print('applying Blur distortion at %d %% noise'%(int(self.noise_level_cam*100)))
            return self.blur(img)

        if deform_type == 'High_exposure':
            if args.verbose:
                print('applying High_exposure distortion at %d %% noise'%(int(self.noise_level_cam*100)))
            return self.high_exposure(img)

        if deform_type == 'Low_exposure':
            if args.verbose:
                print('applying Low_exposure distortion at %d %% noise'%(int(self.noise_level_cam*100)))
            return self.low_exposure(img)

        if deform_type == 'Gaussian_noise':
            if args.verbose:
                print('applying Gaussian_noise distortion at %d %% noise'%(int(self.noise_level_cam*100)))
            return self.add_noise(img)

        if deform_type == 'superfish':
            if args.verbose:
                print('applying superfish distortion at %d %% noise'%(int(self.noise_level_cam*100)))
            return self.superfish(img)

        else:
            print('unknown deform type')
            return img

        # # Generating various noises
        # blur_img    = self.blur(img)
        # highexp_img = self.high_exposure(img)
        # lowexp_img  = self.low_exposure(img)
        # noisy_img   = self.add_noise(img)

        # superfish_img = self.superfish(img)   # Deactivated
        # foggy_img = self.s(img)               # Future work


#--------------------------------------------------------------------Main--------------------------------------------------------------------
def create_parser():

    parser = argparse.ArgumentParser()
    
    # nuScenes loading
    parser.add_argument('--nusc_root', type=str, default='./data/nuScenes/', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='mini', help='train/val/test/mini')
    parser.add_argument('--sensor', type=str, nargs='+', default=sensor_list, help='Sensor type (see sensor_list) to focus on')

    # Noise level
    parser.add_argument('--n_level_cam', '-ncam','-cnoise' , type=float, default=0.1, help='Noise level for cams')
    parser.add_argument('--n_level_radar', '-nrad','-rnoise' , type=float, default=0.1, help='Noise level for radars')

    # Output config
    parser.add_argument('--out_root', type=str, default='./data/noisy_nuScenes', help='Noisy output folder')
    parser.add_argument('--no_overwrite', action='store_true', default=False, help='Do not overwrite existing output')

    # Display
    parser.add_argument('--disp_all_data', action='store_true', default=False, help='Display mosaic with camera and radar original info')
    parser.add_argument('--disp_radar', action='store_true', default=False, help='Display original Radar point cloud and new one')
    parser.add_argument('--save_radar_render', action='store_true', default=False, help='Save new original and new Radar point cloud 3D rendering')
    parser.add_argument('--disp_img', action='store_true', default=False, help='Display original Camera image and new one')
    parser.add_argument('--disp_all_img', action='store_true', default=False, help='Display mosaic of camera views')
    parser.add_argument('--gen_lvl_grad_img', action='store_true', default=False, help='generate output files for multiple noise levels')
    parser.add_argument('--gen_paper_img', action='store_true', default=False, help='generate output files for paper (4 outputs)')
    parser.add_argument('--gen_csv', action='store_true', default=False, help='generate csv file out of df (debug)')
    
    # Verbosity level
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity on|off')

    # Other
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--checksum', action='store_true', default=False, help='checks encoding/decoding of files')

    return parser



def check_args(args):
    assert args.split in ['train','val','test','mini'], 'Wrong split type'

    if args.sensor:
        assert all(sensor in sensor_list for sensor in args.sensor), 'Unknown sensor selected: %s'%(args.sensor)  
   
    assert os.path.exists(args.nusc_root), 'Data folder at %s not found'%(args.nusc_root)

    if not os.path.exists(args.out_root):
        mkdir_if_missing(args.out_root)

    print(args)

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Loading nuScenes
    nusc = load_nusc(args.split,args.nusc_root)

    # load arguments for visualizer functions
    visualizer.init_var(nusc,args)

    # Dataset parser for debug
    if args.debug: 
        deformer=deform_data(args)
        parse_nusc_keyframes(nusc, args.sensor, args, deformer)

    # generate noisy dataset
    genDataset(nusc, args.sensor, args)

    exit('end of script')



'''
Running command:
python dataset_handler.py --debug --sensor <SENSOR> -v


Some reading :
https://github.com/nutonomy/nuscenes-devkit/blob/05d05b3c994fb3c17b6643016d9f622a001c7275/python-sdk/nuscenes/utils/data_classes.py#L315
https://forum.nuscenes.org/t/detail-about-radar-data/173/5
https://forum.nuscenes.org/t/radar-vx-vy-and-vx-comp-vy-comp/283/4
https://conti-engineering.com/wp-content/uploads/2020/02/ARS-408-21_EN_HS-1.pdf

# Some papers on the impact of SNR on accuracy :
https://ieeexplore.ieee.org/abstract/document/55565
https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/610920#:~:text=The%20transmitted%20power%20has%20an,target%20%5B7%2C%208%5D.

# Some paper on adverse image distortion:
https://github.com/Gil-Mor/iFish
https://github.com/noahzn/FoHIS
'''
