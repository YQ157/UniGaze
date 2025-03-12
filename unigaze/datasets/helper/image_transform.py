
import cv2
from torchvision import transforms
import numpy as np
import torch

def re_normalize(image_tensor, old='[-1,1]', new='imagenet'):
	"""
	Re-normalizes an image tensor from one normalization scheme to another.
	Args:
		image_tensor (torch.Tensor): Image tensor to be re-normalized.
		old (str): Old normalization scheme. Options: '[-1,1]', 'imagenet'.
		new (str): New normalization scheme. Options: '[-1,1]', 'imagenet'.
	Returns:
		torch.Tensor: Re-normalized image tensor.
	"""
	# Old normalization parameters
	device = image_tensor.device
	if old == '[-1,1]':
		old_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
		old_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
	elif old == 'imagenet':
		old_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
		old_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
	elif old == '[0,1]':
		old_mean = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(device)
		old_std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(device)
	else:
		print('old normalization not implemented')
		raise NotImplementedError
	# New normalization parameters
	if new == '[-1,1]':
		new_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
		new_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
	elif new == 'imagenet':
		new_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
		new_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
	elif new == '[0,1]':
		new_mean = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(device)
		new_std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(device)
	else:
		print('new normalization not implemented')
		raise NotImplementedError
	# Step 1: Denormalize the image tensor using the old mean and std
	denormalized_image = image_tensor * old_std + old_mean
	# Step 2: Normalize the image tensor using the new mean and std
	normalized_image = (denormalized_image - new_mean) / new_std

	return normalized_image






def wrap_transforms(image_transforms_type, image_size):


	if image_transforms_type == 'basic_imagenet':
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]
		return transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				transforms.Normalize(mean=MEAN, std=STD)
			])
	

	else:
		raise NotImplementedError



# def enhance_contrast_clahe(image):
# 	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 	lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# 	lab_planes = list( cv2.split(lab) )
# 	lab_planes[0] = clahe.apply(lab_planes[0])
# 	lab = cv2.merge(lab_planes)
# 	image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
# 	return image
