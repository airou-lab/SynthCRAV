#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import copy
import cv2

root = './data/noisy_nuScenes/examples/presentation/radar_contrast/'


folder = os.path.join(root,'noisy')
print(os.listdir(folder))

for item in os.listdir(folder):
    filename=os.path.join(folder,item)
    img = cv2.imread(filename)
    inverted_img = 255 - copy.deepcopy(img)
    cv2.imwrite(filename, inverted_img)
