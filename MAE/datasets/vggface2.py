import h5py, cv2
import numpy as np
import os
from torch.utils.data import Dataset
from typing import List
from .helper.image_transform import wrap_transforms, crop_resize, compute_l2_norm_large_array


class VGGFace2Dataset(Dataset):
	"""
	the structure of this dataset
		- group1.h5
			- 'face_patch'
			- 'face_gaze'
			- 'face_head_pose'

		- group2.h5
			...
		- groupN.h5
				
	"""
	def __init__(self, 
				dataset_path: str, 
				color_type,
				data_name=None, 
				gruops_to_use: List[str] = None, 
				image_size:int=224,
				transform_type='basic',
				num_images_person=None,
				whether_crop_resize=True,
				limit_head_pose=None,
				 ):

		super().__init__()
		self.dataset_path = dataset_path
		self.hdfs = {}
		self.data_name = data_name
		self.image_size = (image_size, image_size)
		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type
		
		if num_images_person is not None:
			used_ratio = num_images_person/100 ## NOTE: this 100 is hardcoding, as the saved data has 100 images per person
		else:
			used_ratio = 1
		#### -------------------------------------------------------- read the h5 files ------------------------------------------------------- 
		self.selected_keys = [k for k in gruops_to_use]
		assert len(self.selected_keys) > 0
		self.file_paths = [os.path.join(self.dataset_path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.dataset_path, self.selected_keys[num_i])
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.dataset_path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode
		####----------------------------------------------------------------------------------------------------------------------------------- 

		self.idx_to_kv = []
		self.key_idx_dict = {}
		
		for num_i in range(0, len(self.selected_keys)):
			key_name = self.selected_keys[num_i].split('.')[0]
			n = self.hdfs[num_i]['face_patch'].shape[0]
			if limit_head_pose is None:
				indices = np.arange(0, n)
			else:
				head_pose = self.hdfs[num_i]['face_head_pose']
				indices = compute_l2_norm_large_array(head_pose, limit_head_pose)

			if used_ratio == 1:
				n_ids = indices
			else:
				used_indices = int(used_ratio * n)
				if len(indices) <= used_indices:
					n_ids = indices
				else:
					n_ids = np.random.choice(indices, used_indices, replace=False)
			self.idx_to_kv += [(num_i, i) for i in n_ids]
			self.key_idx_dict[key_name] = [ i for i in n_ids ]

		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.transform = wrap_transforms(transform_type, image_size=image_size)
		self.__hdfs = None
		self.hdf = None
	

		self.whether_crop_resize = whether_crop_resize
		


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
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
			image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8)	)
		return image
	
	def __getitem__(self, idx):
		# Pick entry a and b from same person
		num_i, data_idx = self.idx_to_kv[idx]
		## num_i: the index of the group
		self.hdf = self.archives[num_i]
		assert self.hdf.swmr_mode

		image = self.hdf['face_patch'][data_idx]
		head_label = self.hdf['face_head_pose'][data_idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0,0]).astype('float')
		
		if np.random.rand() > 0.5 and self.whether_crop_resize:
			random_rate = np.random.uniform(low = 0.75, high = 0.9) # random center crop rate
			image = crop_resize(image, random_rate) 
		entry = {
			'image': self.preprocess_image(image),
			'gaze': np.array([0,0]).astype('float'),
			'head': head_label,
			# 'key': num_i,
			# 'index': data_idx
		}
		
		return entry
	
