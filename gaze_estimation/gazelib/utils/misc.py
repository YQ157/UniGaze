
import os, sys
import csv
import yaml
import argparse
import glob
import os.path as osp
from glob import glob
import cv2
import numpy as np
import matplotlib.path as mplPath



def read_resize_blur(img_path, roi_size):
	background_image = cv2.imread(img_path)
	background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)
	background_image = cv2.resize(background_image, roi_size, interpolation=cv2.INTER_AREA)
	background_image = cv2.blur(background_image, (20,20))
	background_image = np.array(background_image) / 255.0
	return background_image



def get_mask_index(lm_gt, vertices, lm_3d, crop_params):
	'''
    args:
        lm_gt: ground truth 2D landmarks
        vertices: 3D face vertices from reconstruction
        lm_3d: 3D landmarks from reconstruction
        crop_params: cropping matrix from reconstruction
    return:
        mask_index: the index of 3D face vertices that are inside the face region
	'''
	lm_temp = np.c_[ lm_gt, np.ones(68) ] 
	cropped_landmarks =  np.matmul(lm_temp, np.transpose(crop_params))
	outline = cropped_landmarks[[*range(17), *range(26,16,-1)],:2]
	bbPath = mplPath.Path(outline)
	face_mask = bbPath.contains_points(vertices[:,:2]) 
	lm_jaw = lm_3d[5,2] # the 5th index is the landmark near the jaw of face, only want the closer part than it
	jaw_mask = np.where(vertices[:,2]<lm_jaw, True, False)
	new_mask = face_mask * jaw_mask # (n_vertices,)
	mask_index = np.where(new_mask==True)[0]
	return mask_index



def tuple_to_str(t):
	return ','.join(map(str, t))
def str_to_tuple(s):
	return tuple(map(float, s.split(',')))




def parse_roi_box_from_landmark(pts, cam=None):
	from math import sqrt
	"""calc roi box from landmark"""
	bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
	
	center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
	
	temp_radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

	if cam in [ 'cam14']: ## move the center down a little
		center[1] += 0.3 * temp_radius
	
	if cam in ['cam04', 'cam05', 'cam12']: ## move the center down a little-little
		center[1] += 0.2 * temp_radius
	

	if cam in ['cam12', 'cam13', 'cam14']: ## the face in these cameras are small, so use smaller radius
		radius =  1.5 * temp_radius
	elif cam in ['cam11', 'cam15']: ## the face in these cameras are large, so use larger radius
		radius = 1.65 * temp_radius
	else:
		radius = 1.5 * temp_radius
	

	bbox = [center[0] - radius,   # left
			center[1] - 1.5 * radius,   # up
			center[0] + radius,   # right
			center[1] + 0.5 * radius ]  # bottom

	llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
	center_x = (bbox[2] + bbox[0]) / 2
	center_y = (bbox[3] + bbox[1]) / 2

	roi_box = [0] * 4
	roi_box[0] = center_x - llength / 2
	roi_box[1] = center_y - llength / 2
	roi_box[2] = roi_box[0] + llength
	roi_box[3] = roi_box[1] + llength

	return roi_box

def crop_img(img, roi_box):
	h, w = img.shape[:2]

	sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
	dh, dw = ey - sy, ex - sx
	if len(img.shape) == 3:
		res = np.zeros((dh, dw, 3), dtype=np.uint8)
	else:
		res = np.zeros((dh, dw), dtype=np.uint8)
	if sx < 0:
		sx, dsx = 0, -sx
	else:
		dsx = 0

	if ex > w:
		ex, dex = w, dw - (ex - w)
	else:
		dex = dw

	if sy < 0:
		sy, dsy = 0, -sy
	else:
		dsy = 0

	if ey > h:
		ey, dey = h, dh - (ey - h)
	else:
		dey = dh

	res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
	return res


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


