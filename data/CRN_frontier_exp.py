#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------
import os
import shutil
import random
import subprocess

def cleanup():
	generated_folders = ['depth_gt','radar_bev_filter','radar_pv_filter']
	generated_files = ['nuscenes_infos_val.pkl']
	for folder in generated_folders:
		if os.path.exists('./data/frontier_nuScenes/'+folder):
			shutil.rmtree('./data/frontier_nuScenes/'+folder)
	
	for file in generated_files:
		if os.path.exists('./data/frontier_nuScenes/'+file):
			os.remove('./data/frontier_nuScenes/'+file)




sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT','LIDAR_TOP']

og_nusc_root = './data/og_nuScenes/'
noisy_nusc_root = './data/noisy_nuScenes/'
frontier_nuscenes_root = './data/frontier_nuScenes/'

if not os.path.exists(frontier_nuscenes_root):
	os.mkdir(frontier_nuscenes_root)

frontier_nuscenes = os.path.join(frontier_nuscenes_root,'samples')	
if not os.path.exists(frontier_nuscenes):
	os.mkdir(frontier_nuscenes)
else:
	[os.remove(os.path.join(frontier_nuscenes,item)) for item in os.listdir(frontier_nuscenes)]
	os.rmdir(frontier_nuscenes)
	os.mkdir(frontier_nuscenes)

if 'maps' not in os.listdir(frontier_nuscenes_root):
	shutil.copytree(os.path.join(og_nusc_root,'maps'),os.path.join(frontier_nuscenes_root,'maps'))

if 'v1.0-mini' not in os.listdir(frontier_nuscenes_root):
	shutil.copytree(os.path.join(og_nusc_root,'v1.0-mini'),os.path.join(frontier_nuscenes_root,'v1.0-mini'))

if 'v1.0-test' not in os.listdir(frontier_nuscenes_root):
	shutil.copytree(os.path.join(og_nusc_root,'v1.0-test'),os.path.join(frontier_nuscenes_root,'v1.0-test'))

frontier_nuscenes = os.path.join(frontier_nuscenes_root,'samples')

if not os.path.exists('./frontier_outputs'):	os.mkdir('./frontier_outputs')

for item in os.listdir(frontier_nuscenes):
	os.remove(os.path.join(frontier_nuscenes,item))

cleanup()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# STEP 1 : ORIGINAL DATA
print()
print(80*'#','ORIGINAL DATA:',80*'#')
for sensor in sensor_list:
	print('current sensor:',sensor)

	symlink_path = os.path.join(frontier_nuscenes,sensor)
	symlink_target = os.path.relpath(os.path.join(og_nusc_root, 'samples', sensor), frontier_nuscenes)


	print('linking', symlink_target, '@', symlink_path)
	os.symlink(symlink_target, symlink_path)

# --> test CRN here
process_return_code = subprocess.run(["bash", "-c", "python scripts/gen_info.py && \
										python scripts/gen_depth_gt.py && \
										python scripts/gen_radar_bev.py && \
										python scripts/gen_radar_pv.py && \
										python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1"])
if process_return_code.returncode!=0:
	print('graceful exit')
	exit()

output_folder = './frontier_outputs/original'
try:
	shutil.copytree("outputs/det/CRN_r50_256x704_128x128_4key",output_folder)
except FileExistsError:
	pass
except:
	exit()


# STEP 2 : CAM SYNTH DATA
print()
print(80*'#','CAM SYNTH:',80*'#')
for noise_lvl in range(10,110,10):
	print(10*'*','CURRENT NOISE LEVEL:',noise_lvl,10*'*')
	cleanup()

	for sensor in sensor_list:
		if 'CAM' in sensor:
			symlink_path = os.path.join(frontier_nuscenes,sensor)

			# delete previous symlinks
			if sensor in os.listdir(frontier_nuscenes):	# not using os.path.exists() in case the symlink is broken
				os.remove(os.path.join(symlink_path))
				print('removed',symlink_path)

			# new link
			noise_type = random.choice(['Blur','Gaussian_noise','High_exposure','Low_exposure'])
			symlink_target = os.path.relpath(os.path.join(noisy_nusc_root, 'samples', sensor, str(noise_lvl), noise_type), frontier_nuscenes)
			
			print('linking', symlink_target, '@', symlink_path)
			os.symlink(symlink_target, symlink_path)

	# --> test CRN here
	process_return_code = subprocess.run(["bash", "-c", "python scripts/gen_info.py && \
											python scripts/gen_depth_gt.py && \
											python scripts/gen_radar_bev.py && \
											python scripts/gen_radar_pv.py && \
											python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1"])
	if process_return_code.returncode!=0:
		print('graceful exit')
		exit()

	output_folder = './frontier_outputs/cam_frontier_'+str(noise_lvl)
	try:
		shutil.copytree("outputs/det/CRN_r50_256x704_128x128_4key",output_folder)
	except FileExistsError:
		pass
	except:
		exit()


# STEP 3 : RESET
print()
print(80*'#','RESET:',80*'#')
for sensor in sensor_list:
	print('current sensor:',sensor)

	symlink_target = os.path.join('..','..','og_nuScenes','samples',sensor)
	symlink_path = os.path.join(frontier_nuscenes,sensor)

	# delete previous symlinks
	if sensor in os.listdir(frontier_nuscenes):	# not using os.path.exists() in case the symlink is broken
		os.remove(os.path.join(symlink_path))
		print('removed',symlink_path)


	print('linking', symlink_target, '@', symlink_path)
	os.symlink(symlink_target, symlink_path)


# STEP 4 : RADAR SYNTH DATA
print()
print(80*'#','RADAR SYNTH:',80*'#')
for noise_lvl in range(10,110,10):
	cleanup()
	print(10*'*','CURRENT NOISE LEVEL:',noise_lvl,10*'*')
		
	for sensor in sensor_list:
		if 'RADAR' in sensor:
			symlink_path = os.path.join(frontier_nuscenes,sensor)

			# delete previous symlinks
			if sensor in os.listdir(frontier_nuscenes):	# not using os.path.exists() in case the symlink is broken
				os.remove(os.path.join(symlink_path))
				print('removed',symlink_path)

			# new link
			symlink_target = os.path.relpath(os.path.join(noisy_nusc_root, 'samples', sensor, str(noise_lvl)), frontier_nuscenes)

			print('linking', symlink_target, '@', symlink_path)
			os.symlink(symlink_target, symlink_path)

	## --> test CRN here
	process_return_code = subprocess.run(["bash", "-c", "python scripts/gen_info.py && \
											python scripts/gen_depth_gt.py && \
											python scripts/gen_radar_bev.py && \
											python scripts/gen_radar_pv.py && \
											python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1"])
	if process_return_code.returncode!=0:
		print('graceful exit')
		exit()

	output_folder = './frontier_outputs/radar_frontier_'+str(noise_lvl)
	try:
		shutil.copytree("outputs/det/CRN_r50_256x704_128x128_4key",output_folder)
	except FileExistsError:
		pass
	except:
		exit()