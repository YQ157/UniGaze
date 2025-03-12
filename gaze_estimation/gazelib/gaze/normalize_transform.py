"""
transform_norm change the image from normalization_1 to normalization_2
"""

import os
import os.path as osp
import argparse
import math
import numpy as np
import imageio
import scipy.io
import glob
import cv2
import matplotlib.pyplot as plt
import h5py
import yaml
import time

from tqdm import tqdm




def resize_landmarks(lm68, focal_norm, src_size, tgt_size):
	'''
	resize landmarks if the image is resized
	'''
	# S = np.eye(3)
	cam_src = np.array([
		[focal_norm, 0, src_size[0]/2],
		[0, focal_norm, src_size[1]/2],
		[0, 0, 1.0],
	])

	S = np.array([ # scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, src_size[0]/tgt_size[0]],
	])


	cam_tgt = np.array([
		[focal_norm, 0, tgt_size[0]/2],
		[0, focal_norm, tgt_size[1]/2],
		[0, 0, 1.0],
	])
	
	W = cam_tgt @ S @ np.linalg.inv(cam_src)
	num_point = lm68.shape[0]
	landmarks_warped = cv2.perspectiveTransform(lm68.reshape(-1,1,2).astype('float32'), W)
	landmarks_warped = landmarks_warped.reshape(num_point, 2)
	return landmarks_warped





def transform_norm(image_norm_1, original_norm_config, target_norm_config):
	
	focal_1 = original_norm_config['focal_norm'] # focal length of normalized camera
	distance_1 = original_norm_config['distance_norm']  # normalized distance between eye and camera
	roi_size_1 = original_norm_config['roi_size']  # size of cropped eye image
	cam1 = np.array([
			[focal_1, 0, roi_size_1[0]/2],
			[0, focal_1, roi_size_1[1]/2],
			[0, 0, 1.0],
		])

	focal_2 = target_norm_config['focal_norm'] # focal length of normalized camera
	distance_2 = target_norm_config['distance_norm']  # normalized distance between eye and camera
	roi_size_2 = target_norm_config['roi_size']  # size of cropped eye image
	cam2 = np.array([
			[focal_2, 0, roi_size_2[0]/2],
			[0, focal_2, roi_size_2[1]/2],
			[0, 0, 1.0],
		])
	z_scale = distance_2/distance_1
	S = np.array([ # scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, z_scale],
	])

	W = np.dot(np.dot(cam2, S), np.dot(np.eye(3), np.linalg.inv(cam1))) # transformation matrix
	image_norm_2 = cv2.warpPerspective(image_norm_1, W, roi_size_2) # image normalization
	return image_norm_2


# def plot_gaze_3d(landmarks1, face_center1, landmarks2, face_center2, gc, save_dir):
# 	# landmarks = landmarks.reshape(-1, 3)
# 	face_center1 = face_center1.reshape(1,3)
# 	face_center2 = face_center2.reshape(1,3)

# 	gc = gc.reshape(1,3)

# 	fig = plt.figure()
# 	ax = fig.add_subplot(projection='3d')
# 	# ax.set_aspect('equal')
# 	ax.set_box_aspect((1,1,1))

# 	def plot_one_group(landmarks, face_center, gc, color):
# 		for i in range(landmarks.shape[0]):
# 			x, y, z = landmarks[i,0], landmarks[i,1], landmarks[i, 2]
# 			ax.scatter(x, y, z, s=5)
# 			ax.text(x, y, z, i)

# 		ax.scatter(face_center[0,0], face_center[0,1], face_center[0,2], s=5)
		
# 		vector = gc - face_center
# 		vector /= np.linalg.norm(vector) 
# 		vector *= 50
# 		gc = face_center + vector

# 		ax.plot( [face_center[0,0], gc[0, 0]],
# 				[face_center[0,1], gc[0, 1]],
# 				[face_center[0,2], gc[0, 2]], zdir='z', c=color, linewidth=3)

# 	plot_one_group(landmarks1, face_center1, gc, 'red')
# 	plot_one_group(landmarks2, face_center2, gc, 'green')
# 	gaze_vector_1 = gc.reshape((3,1)) - face_center1.reshape(3,1) # (3,1)
# 	gaze_vector_2 = gc.reshape((3,1)) - face_center2.reshape(3,1) # (3,1)
	
# 	diff=vector_angle_difference(gaze_vector_1, gaze_vector_2)

# 	ax.set_title("the angle difference: {:.2f} degree".format( diff[0]))
# 	# ax.set_xlim3d(-0.8, 0.8)
# 	# ax.set_ylim3d(-0.8, 0.8)
# 	# ax.set_zlim3d(-0.1, 1.5)
# 	ax.set_xlabel('$X$', fontsize=10)
# 	ax.set_ylabel('$Y$', fontsize=10)
# 	ax.set_zlabel('$Z$', fontsize=10)
# #     ax.invert_xaxis()
# 	plt.show()
# 	plt.savefig(save_dir + '/gaze_vector.jpg')
# 	plt.close()
# 	plt.clf()



# def vector_angle_difference(vector1, vector2):
# 	a = vector1.reshape(-1,3)
# 	b = vector2.reshape(-1,3)

# 	ab = np.sum(np.multiply(a, b), axis=1)
# 	a_norm = np.linalg.norm(a, axis=1)
# 	b_norm = np.linalg.norm(b, axis=1)

# 	# Avoid zero-values (to avoid NaNs)
# 	a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
# 	b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

# 	similarity = np.divide(ab, np.multiply(a_norm, b_norm))

# 	return np.arccos(similarity) * 180.0 / np.pi

# def transform(frame_path, original_norm_config, target_norm_config, save_path, use_50=False):
	
# 	focal_1 = original_norm_config['focal_norm'] # focal length of normalized camera
# 	distance_1 = original_norm_config['distance_norm']  # normalized distance between eye and camera
# 	roi_size_1 = original_norm_config['roi_size']  # size of cropped eye image
# 	cam1 = np.array([
# 			[focal_1, 0, roi_size_1[0]/2],
# 			[0, focal_1, roi_size_1[1]/2],
# 			[0, 0, 1.0],
# 		])

# 	focal_2 = target_norm_config['focal_norm'] # focal length of normalized camera
# 	distance_2 = target_norm_config['distance_norm']  # normalized distance between eye and camera
# 	roi_size_2 = target_norm_config['roi_size']  # size of cropped eye image
# 	cam2 = np.array([
# 			[focal_2, 0, roi_size_2[0]/2],
# 			[0, focal_2, roi_size_2[1]/2],
# 			[0, 0, 1.0],
# 		])
# 	z_scale = distance_2/distance_1
# 	S = np.array([ # scaling matrix
# 		[1.0, 0.0, 0.0],
# 		[0.0, 1.0, 0.0],
# 		[0.0, 0.0, z_scale],
# 	])


# 	image_list = sorted(glob.glob(frame_path + '/' + '*.JPG'))
# 	print(" image_list: ", image_list)
# 	for input_path in image_list:
# 		img_name = os.path.basename(input_path) 
# 		frame = input_path.split('/')[-2]
# 		subject = input_path.split('/')[-3]

# 		img_1 = cv2.imread(input_path)

# 		# #  Data Normalization
# 		W = np.dot(np.dot(cam2, S), np.dot(np.eye(3), np.linalg.inv(cam1))) # transformation matrix
# 		img_2 = cv2.warpPerspective(img_1, W, roi_size_2) # image normalization
		
# 		os.makedirs(os.path.join(save_path, 'sample', frame), exist_ok=True)
# 		cv2.imwrite( os.path.join(save_path,'sample',  frame, img_name), img_2 )
