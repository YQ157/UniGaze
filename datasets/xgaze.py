import os,random
import numpy as np
import h5py
import cv2
from typing import List
from torch.utils.data import Dataset
from .helper.image_transform import wrap_transforms

class XGazeDataset(Dataset):
	def __init__(self, 
				dataset_path: str, 
				color_type,
				images_per_frame,
				keys_to_use: List[str] = None, 
				data_name=None, 
				image_size:int=224, 
				transform_type='basic_imagenet', ## <--- modified
				image_key='face_patch',
				gaze_key='face_gaze',
				camera_random=None,
				frame_tag=[0,1000],
				seed=0,
				):

		self.path = dataset_path
		self.hdfs = {}
		self.data_name = data_name
		self.images_per_frame = images_per_frame

		print('images_per_frame: ', images_per_frame)
		self.image_key = image_key
		self.gaze_key = gaze_key
		self.image_size = (image_size, image_size)
		random.seed(seed)
		
		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type
		self.cameras_idx = list(range(self.images_per_frame))
		self.camera_random = camera_random 

		#### -------------------------------------------------------- read the h5 files ------------------------------------------------------- 
		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0
		self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode
		####----------------------------------------------------------------------------------------------------------------------------------- 

		
		self.idx_to_kv = []
		self.key_idx_dict = {} ## this is for reading the second sample from the same person
		for num_i in range(0, len(self.selected_keys)):
			this_sub = self.selected_keys[num_i].split('.')[0]
			n = self.hdfs[num_i][image_key].shape[0] 

			if type(frame_tag) == list:
				self.start_frame, self.end_frame = frame_tag
			elif frame_tag == 'all':
				self.start_frame, self.end_frame = 0, 10000
			else:
				raise ValueError("frame_tag should be either a list of integers or str 'all' ")
			start_idx = min(n, self.start_frame * self.images_per_frame)
			end_idx =  min(n, self.end_frame  * self.images_per_frame)

			if self.camera_random is None:
				self.idx_to_kv +=  [(num_i, i) for i in range(start_idx, end_idx) if (i % self.images_per_frame ) in self.cameras_idx ]
				self.key_idx_dict[this_sub] = [ i for i in range(start_idx, end_idx) if (i % self.images_per_frame ) in self.cameras_idx ]
			else:
				for frame in range(start_idx // self.images_per_frame, end_idx // self.images_per_frame):
					frame_start_idx = frame * self.images_per_frame
					frame_end_idx = frame_start_idx + self.images_per_frame

					# Randomly select self.images_per_frame camera indices for this frame
					random_cameras_idx = random.sample(range(self.images_per_frame), self.camera_random)
					self.idx_to_kv += [(num_i, i) for i in range(frame_start_idx, frame_end_idx) if (i % self.images_per_frame) in random_cameras_idx]
					self.key_idx_dict.setdefault(this_sub, []).extend(
						[i for i in range(frame_start_idx, frame_end_idx) if (i % self.images_per_frame) in random_cameras_idx]
					)

		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.transform = wrap_transforms(transform_type, image_size=image_size)
		self.__hdfs = None
		self.hdf = None


	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

	
	@property
	def archives(self):
		if self.__hdfs is None: # lazy loading here!
			self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
		return self.__hdfs

	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			image = image[..., ::-1]
		if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
			image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

		image = self.transform( image.astype(np.uint8) )
		return image

	def __getitem__(self, index):
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		assert self.hdf.swmr_mode
		image = self.hdf[self.image_key][idx, :]
		gaze_label = self.hdf[self.gaze_key][idx].astype('float') if self.gaze_key in self.hdf else np.array([0,0]).astype('float')
		head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0,0]).astype('float')
		
		entry = {
			'image': self.preprocess_image(image),
			'gaze': gaze_label,
			'head': head_label,
			'key': key,
			'index':index
		}

		return entry


