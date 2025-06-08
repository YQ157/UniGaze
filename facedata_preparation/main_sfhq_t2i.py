import os, sys
import os.path as osp
import argparse
import numpy as np
import glob
import cv2
import h5py
import time
import zipfile
from collections import defaultdict
from omegaconf import OmegaConf
from datetime import datetime
import face_alignment
from rich.progress import track

from util_func import rad_to_degree, draw_sns, grid_images, write_error, set_dummy_camera_model
from landmarks_func import get_landmarks_from_image
unigaze_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, unigaze_root)
from unigaze.gazelib.label_transform import lm68_to_50, get_eye_nose_landmarks 
from unigaze.gazelib.gaze.normalize import normalize, estimateHeadPose
from unigaze.gazelib.utils.h5_utils import add, to_h5
from unigaze.gazelib.label_transform import get_eye_nose_landmarks, get_face_center_by_nose
sys.path.remove(unigaze_root)




def main():



	
	error_log = osp.join(output_dir, 'error_log.txt')
	
	sample_dir = osp.join(output_dir, 'sample')
	os.makedirs(sample_dir, exist_ok=True)

	zip_file_path = input_path

	# Open the zip file
	with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
		# List all files in the zip
		zip_contents = zip_file.namelist()
		image_files = [f for f in zip_contents if f.endswith('.jpg') and ('images/images/' in f) ]
		# Sort the list of image files
		image_files = sorted(image_files)
		print(" number of images in images/images: ", len(image_files))


		img_pool = []
		for i in track(range(0, len(image_files)) ):
			img_path = image_files[i]

			print(f"  - Reading {img_path}")
			with zip_file.open(img_path) as file:
				image_data = file.read()
				np_arr = np.frombuffer(image_data, np.uint8) # Convert binary data to a NumPy array
				image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decode the image array into an OpenCV image


			
			landmarks = get_landmarks_from_image( image, img_name=img_path,
								fa=fa, error_log=error_log, resize_factor=resize_factor)
			if landmarks is None:
				continue

			lm_gt = landmarks.copy()
			camera_matrix, camera_distortion = set_dummy_camera_model( image = image)
			##  Data Normalization
			## --------------------------------------------  estimate head pose --------------------------------------------
			face_model_load = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), 'face_model.txt') )
			''' Use 50 landmarks to estimate head pose, or only use 6 landmarks '''
			if use_50:
				hr, ht = estimateHeadPose(lm68_to_50(lm_gt).astype(float).reshape(50, 1, 2) , face_model_load.reshape(50, 1, 3), camera_matrix, camera_distortion, iterate=True)
			else:
				face_model = get_eye_nose_landmarks(face_model_load) # the eye and nose landmarks
				landmarks_sub = get_eye_nose_landmarks(lm_gt) # lm_gt[[36, 39, 42, 45, 31, 35], :]

				hr, ht = estimateHeadPose(landmarks_sub.astype(float).reshape(6, 1, 2), face_model.reshape(6, 1, 3), camera_matrix, camera_distortion, iterate=True)
			# # -------------------------------------------------------------------------------------------------------------------
			# compute estimated 3D positions of the landmarks
			ht = ht.reshape((3,1))
			hR = cv2.Rodrigues(hr)[0] # rotation matrix
			face_center_by_nose, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)
			
			# -------------------------------------------- normalize image --------------------------------------------
			img_face, R, hR_norm, gaze_normalized, landmarks_norm, W = normalize(image, lm_gt, focal_norm, distance_norm, roi_size, face_center_by_nose, hr, ht, camera_matrix, gc=None)
			hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
			hr_norm_l2norm = np.linalg.norm(hr_norm)

			img_save = cv2.resize(img_face, (image_save_size, image_save_size), interpolation = cv2.INTER_AREA)
			landmarks_norm_save = landmarks_norm * image_save_size / roi_size[0]

			img_vis = img_save.copy()
			hr_str = " hr: {:.2f}, {:.2f}, {:.2f}".format(hr[0,0], hr[1,0], hr[2,0])
			hr_norm_str = " hr_norm: {:.2f}, {:.2f}".format(hr_norm[0], hr_norm[1])
			cv2.putText(img_vis, hr_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(img_vis, hr_norm_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(img_vis, "{:.2f}".format(hr_norm_l2norm), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


			
			if (np.abs( hr_norm[0] ) > 1.0 and np.abs( hr_norm[1] ) > 1.0) or (hr_norm_l2norm > 1.1):
				print(f"  - Skipping {img_path} because headpose is too large: {hr_norm}")
				# continue
				img_pool.append(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
			else:
				img_pool.append(img_vis)
				to_write = {}
				add(to_write, 'face_patch', img_save)
				add(to_write, 'face_gaze', np.array([0,0]))
				add(to_write, 'face_head_pose', hr_norm)
				add(to_write, 'face_mat_norm', R)
				add(to_write, 'landmarks_norm', landmarks_norm_save )
				to_h5(to_write, save_path)
			

			
			if len(img_pool) == 25:
				grid_shape = (5, 5)
				# Create the image grid
				grid = grid_images(img_pool, grid_shape)
				grid = cv2.resize( grid, dsize=None, fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)
				cv2.imwrite(osp.join(sample_dir, f'sample_{i+1}.jpg'), grid)
				img_pool = []



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str,help='name of the original dataset')
	parser.add_argument('--output_dir', type=str,help='path of the output directory')
	config, _ = parser.parse_known_args()

	### Normalization parameters
	focal_norm = 960 # focal length of normalized camera
	distance_norm = 300  # normalized distance between eye and camera
	roi_size = (448, 448)  # size of cropped eye image

	
	input_path = config.input_path
	output_dir = config.output_dir; os.makedirs(output_dir, exist_ok=True)
	OmegaConf.save(
		{'normalization': { "focal_norm":focal_norm, "distance_norm":distance_norm, "roi_size":roi_size, } },
		  output_dir + '/norm_config.yaml'
		)
	
	
	
	resize_factor = 0.25 ## for faster face landmark detection
	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	except:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


	image_save_size = 224
	use_50=False

	save_path = os.path.join(output_dir, 'sfhqt2i_224.h5')

	main()

