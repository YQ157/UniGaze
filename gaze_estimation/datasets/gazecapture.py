import os
import numpy as np
import h5py
import cv2
from torch.utils.data import Dataset
from typing import List
from omegaconf import OmegaConf, listconfig
from .helper.image_transform import wrap_transforms


class GazeCaptureDataset(Dataset):
	def __init__(self, 
				dataset_path: str, 
				color_type,
				keys_to_use: List[str] = None, 
				data_name=None, 
				image_size:int=224,  ## <--- 
				transform_type='basic_imagenet', ## <--- modified
				image_key='face_patch',
				gaze_key='face_gaze',
				sample_rate_use=1,
				):
		
		self.transform = wrap_transforms(transform_type, image_size=image_size)

		self.path = dataset_path
		self.hdfs = {}
		self.data_name = data_name
		self.image_key = image_key
		self.gaze_key = gaze_key

		self.image_size = (image_size, image_size)

		self.sample_rate_use = sample_rate_use

		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type
		self.selected_keys = [ k for k in keys_to_use]
		assert len(self.selected_keys) > 0
		
		self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode


		self.build_idx_to_kv()
		

		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.__hdfs = None
		self.hdf = None

	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

	def build_idx_to_kv(self):
		self.idx_to_kv = []
		self.key_idx_dict = {}
		for num_i in range(0, len(self.selected_keys)):
			this_sub = self.selected_keys[num_i].split('.')[0]
			n = self.hdfs[num_i][self.image_key].shape[0] 
			if self.sample_rate_use > 1:
				indices = np.arange(0, n, self.sample_rate_use)
			else:
				indices = np.arange(0, n)
			self.idx_to_kv += [(num_i, i) for i in indices ]
			self.key_idx_dict[this_sub] = [ i for i in indices ]

	@property
	def archives(self):
		if self.__hdfs is None: # lazy loading here!
			self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
		return self.__hdfs


	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			image = image[..., ::-1]
		image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8)		)
		return image

	def __getitem__(self, index):
		
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]

		# self.hdf = h5py.File(os.path.join(self.path, self.selected_keys[key]), 'r', swmr=True)
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

# class GazeCaptureDatasetSubset(GazeCaptureDataset):
# 	def __init__(self, images_per_person=None, **kwargs):
# 		self.images_per_person = images_per_person
# 		super().__init__(**kwargs)

# 	def build_idx_to_kv(self):
# 		self.idx_to_kv = []
# 		self.key_idx_dict = {}
# 		for num_i in range(0, len(self.selected_keys)):
# 			this_sub = self.selected_keys[num_i].split('.')[0]
# 			n = self.hdfs[num_i][self.image_key].shape[0] 
# 			if self.images_per_person is not None:
# 				n = min(n, self.images_per_person)
# 			self.idx_to_kv += [(num_i, i) for i in range(n)]
# 			self.key_idx_dict[this_sub] = [ i for i in range(n)]
