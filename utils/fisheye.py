import imageio
import numpy as np
from math import sqrt
import sys
import os
import cv2

# From https://github.com/Gil-Mor/iFish, modified


# def resize_input_img(img,distortion_coefficient):

#     corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
#     max_displacement = 0
    
#     for x, y in corners:
#         rd = sqrt(x**2 + y**2)
#         x_new, y_new = get_fish_xn_yn(x, y, rd, distortion_coefficient)
#         print('x,y:',x,y)
#         print('x_new,y_new:',x_new,y_new)
#         # Check how far the corner moved from its original position
#         displacement = sqrt(x_new**2 + y_new**2)
#         max_displacement = max(max_displacement, displacement)

#     scale_factor = max_displacement
#     new_w, new_h = int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)
#     print('displacement:',displacement)
#     print('max_displacement:',max_displacement)
#     print('img.shape',img.shape)
#     print('new_w,new_h:',new_w,new_h)
#     resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
#     return resized_img

# def crop_img(x,y,warped_img,og_img):
#     output_img = np.zeros_like(og_img)
#     h,w = warped_img.shape[:2]
#     dx = int((w-x)/2)
#     dy = int((h-y)/2)

#     print(w,h,dx,dy)

#     output_img = warped_img[dy:900+dy,dx:dx+1600,:]
#     print(output_img.shape)

#     return output_img


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))


def generate_fisheye_dist(img, distortion_coefficient,resize=False):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # resized_img = resize_input_img(img,distortion_coefficient)

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    h, w = img.shape[:2]
    # if len(resized_img.shape) == 2:
    #     # Duplicate the one BW channel twice to create Black and White
    #     # RGB image (For each pixel, the 3 channels have the same value)
    #     bw_channel = np.copy(resized_img)
    #     resized_img = np.dstack((resized_img, bw_channel))
    #     resized_img = np.dstack((resized_img, bw_channel))
    # if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
    #     # print("RGB to RGBA")
    #     resized_img = np.dstack((resized_img, np.full((h, w), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - h)/h), float((2*y - w)/w)

            # get xn and yn distance from normalized center
            rd = sqrt(xnd**2 + ynd**2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*h)/2), int(((ydu + 1)*w)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < h and 0 <= yu and yu < w:
                dstimg[x][y] = img[xu][yu]

    # if resize:
    #     print(dstimg.shape)
    #     dstimg_resized = resize_input_img(dstimg,distortion_coefficient)
    #     dstimg_resized = crop_img(1600,900,dstimg_resized,img)
    #     return dstimg_resized.astype(np.uint8)

    return dstimg.astype(np.uint8)
