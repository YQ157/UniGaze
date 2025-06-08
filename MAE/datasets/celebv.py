from tqdm import tqdm
import h5py, cv2
import numpy as np
import os
from torch.utils.data import Dataset
from typing import List

from .helper.image_transform import wrap_transforms, crop_resize, compute_l2_norm_large_array




class CelebVDataset(Dataset):
	def __init__(self, 
				dataset_path: str, 
				color_type,
				keys_to_use: List[str] = None, 
				data_name=None, 
				image_size:int=224,
				transform_type='basic',
				sample_rate_use=1,
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
		self.transform = wrap_transforms(transform_type, image_size=image_size)

		self.sample_rate_use = sample_rate_use
		self.limit_head_pose = limit_head_pose
		#### -------------------------------------------------------- read the h5 files ------------------------------------------------------- 
		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0
		self.file_paths = [os.path.join(self.dataset_path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.dataset_path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.dataset_path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode
		####----------------------------------------------------------------------------------------------------------------------------------- 

		self.build_idx_to_kv()
		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.__hdfs = None
		self.hdf = None


		self.whether_crop_resize = whether_crop_resize
	def build_idx_to_kv(self):
		self.idx_to_kv = []
		self.key_idx_dict = {}

		for num_i in range(0, len(self.selected_keys)):
			p_key = self.selected_keys[num_i].split('.')[0]  ##p00
			n = self.hdfs[num_i][self.image_key].shape[0] 

			if self.limit_head_pose is not None:
				head_pose = self.hdfs[num_i]['face_head_pose'][:]
				indices = compute_l2_norm_large_array(head_pose, self.limit_head_pose)
			else:
				indices = np.arange(0, n)
				
			if self.sample_rate_use > 1:
				# n = min(n, self.sample_rate_use)
				indices = indices[::self.sample_rate_use]
			else:
				indices = indices # np.arange(0, n)
			
			# if self.index_specific is not None:
			# 	indices = self.index_specific
			self.idx_to_kv += [(num_i, i) for i in indices]
			self.key_idx_dict[p_key] = [i for i in indices]


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
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image[..., ::-1]
		if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
			image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8)		)
		return image
	
	def __getitem__(self, index):
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		image = self.hdf['face_patch'][idx]
		head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0,0]).astype('float')
		

		if np.random.rand() > 0.5 and self.whether_crop_resize:
			random_rate = np.random.uniform(low = 0.75, high = 0.9) # random center crop rate
			image = crop_resize(image, random_rate) 


		entry = {
			'image': self.preprocess_image(image),
			'gaze': np.array([0,0]).astype('float'), ## dummy
			'head': head_label,
			# 'key': idx,
			# 'index':index
		}
		return entry
	


