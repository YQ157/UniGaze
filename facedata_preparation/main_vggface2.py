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
import zipfile, tarfile, tempfile
from tqdm import tqdm
from util_func import rad_to_degree, draw_sns, grid_images, write_error, get_largest_face
from landmarks_func import get_landmarks_from_image

unigaze_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, unigaze_root)
from unigaze.gazelib.label_transform import lm68_to_50, get_eye_nose_landmarks 
from unigaze.gazelib.gaze.normalize import normalize, estimateHeadPose
from unigaze.gazelib.utils.h5_utils import add, to_h5
from unigaze.gazelib.label_transform import get_eye_nose_landmarks, get_face_center_by_nose
sys.path.remove(unigaze_root)




def is_within_bounds(landmarks, img_width, img_height):
	for (x, y) in landmarks:
		if x < 0 or x >= img_width or y < 0 or y >= img_height:
			return False
	return True




def main():
	error_log = osp.join( osp.dirname(output_dir), 'error_log.txt')

	save_path = osp.join(output_dir, group_name + '.h5')

	person_count = {}
	for person_name in file_paths:
		file_paths_person = file_paths[person_name]
		for file_path in file_paths_person:
			person_path = os.path.join(input_dir, person_name)
			if not os.path.exists(person_path):
				print(f'{person_path} does not exist')
				break

			print('file_path: ', file_path)
			person_name = file_path.split('/')[0]
			img_name = os.path.basename(file_path)
			img_path = os.path.join(input_dir, file_path)
			if person_name in person_count and person_count[person_name] >= num_per_subject:
				break
			if not os.path.exists(img_path):
				print(f'{img_path} does not exist')
				write_error( f'{img_path} does not exist', error_log)
				continue

			
			image = cv2.imread(img_path)
			h, w = image.shape[:2]
			if h < 200 or w < 200:
				print(f'{img_path} is too small')
				write_error( f'{img_path} is too small', error_log)
				continue
			
			image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			preds = fa.get_landmarks(image_rgb)
			if preds is not None:
				if len(preds)>1:
					landmarks = get_largest_face(preds)
					print(f'{img_name} has more than one face detected')
					write_error( f'{img_name} has more than one face detected', error_log)
				else:
					landmarks = preds[0] # array (68,2)
				landmarks = np.asarray(landmarks)
				img_height, img_width, _ = image.shape
				if not is_within_bounds(landmarks, img_width, img_height):
					print("Landmarks are out of bounds and will be discarded.")
					write_error(  "Landmarks for face are out of bounds and will be discarded.", error_log)
					continue
			else:
				print(f'{img_path} has no face detected')
				write_error( f'{img_path} has no face detected', error_log)
				continue


			lm_gt = landmarks.copy()
			camera_matrix = np.array([
				[ 800, 0.0, h//2],
				[ 0.0, 800, w//2],
				[ 0.0, 0.0, 1.0],
			]) # emprically set a default camera matrix
			camera_distortion = np.zeros((1, 5))


			# Data Normalization
			# -------------------------------------------  estimate head pose --------------------------------------------
			face_model_load = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), 'face_model.txt') )
			if use_50:
				hr, ht = estimateHeadPose(lm68_to_50(lm_gt).astype(float).reshape(50, 1, 2) , face_model_load.reshape(50, 1, 3), camera_matrix, camera_distortion, iterate=True)
			else:
				face_model = get_eye_nose_landmarks(face_model_load) # the eye and nose landmarks
				landmarks_sub = get_eye_nose_landmarks(lm_gt)
				hr, ht = estimateHeadPose(landmarks_sub.astype(float).reshape(6, 1, 2), face_model.reshape(6, 1, 3), camera_matrix, camera_distortion, iterate=True)
			ht = ht.reshape((3,1))
			hR = cv2.Rodrigues(hr)[0] # rotation matrix
			face_center_by_nose, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)
			# -------------------------------------------- normalize image --------------------------------------------
			img_face, R, hR_norm, gaze_normalized, landmarks_norm, W = normalize(image, lm_gt, focal_norm, distance_norm, roi_size, face_center_by_nose, hr, ht, camera_matrix, gc=None)
			hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])

			##  resize the image before writing  ( to save size)
			now_size = img_face.shape[0]
			img_face = cv2.resize(img_face, (image_save_size, image_save_size) , interpolation = cv2.INTER_AREA) 
			landmarks_norm = landmarks_norm * image_save_size / now_size

			
			to_write = {}
			add(to_write, 'face_patch', img_face) # normalized face image
			add(to_write, 'face_gaze', np.array([0,0])) # no gaze information in VGGFace2
			add(to_write, 'face_head_pose', hr_norm) # pitch,yaw for head pose (normalized)
			add(to_write, 'landmarks_norm', landmarks_norm) # the landmarks in the normalized face image
			to_h5(to_write, save_path )

			
			## update the person_count
			if person_name in person_count:
				person_count[person_name] += 1
			else:
				person_count[person_name] = 1

		

		
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str,help='name of the original dataset')
	parser.add_argument('--supp_data', type=str,help='path to the supp file')
	parser.add_argument('--num_per_subject', type=int, default=20, help='number of images per subject to process')
	parser.add_argument('--output_dir', type=str,help='path of the output directory')

	config, _ = parser.parse_known_args()


	num_per_subject = config.num_per_subject
	image_save_size = 224
	use_50=True
	
	### Normalization parameters
	focal_norm = 960 # focal length of normalized camera
	distance_norm = 300  # normalized distance between eye and camera
	roi_size = (448, 448)  # size of cropped eye image

	
	### data path
	input_dir = config.input_dir
	output_dir = config.output_dir; os.makedirs( output_dir, exist_ok=True)


	OmegaConf.save(
		{'normalization': { "focal_norm":focal_norm, "distance_norm":distance_norm, "roi_size":roi_size, },
		'save_size': image_save_size,
   		},
		  output_dir + '/norm_config.yaml'
		)

	
	resize_factor = 0.5 ## for faster face landmark detection
	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	except:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)



	file_paths = OmegaConf.load(config.supp_data)
	group_name = os.path.basename(config.supp_data).split('.')[0]
	main()
	

