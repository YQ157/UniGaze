
import cv2
import numpy as np
import torch
from torchvision import transforms
import tqdm

def wrap_transforms(image_transforms_type, image_size):

	
	if image_transforms_type == 'basic':
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]
		return transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				transforms.Normalize(mean=MEAN, std=STD)
			])
	
	
	elif image_transforms_type == 'jitter':
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]

		brightness = 0.3
		contrast = (0.4, 1.8)
		saturation = 0.2
		hue = 0.15

		color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
		return transforms.Compose([transforms.ToPILImage(),
											transforms.RandomApply([color_jitter], p=0.5),
											transforms.RandomGrayscale(p=0.05),
											transforms.ToTensor(),
											transforms.Normalize(mean=MEAN, std=STD)])
	
	else:
		raise NotImplementedError




def crop_resize( image, random_rate):	
	h, w = image.shape[:2]
	w_crop, h_crop = int(h * random_rate), int(w * random_rate)
	h_start, w_start = (h - h_crop) // 2, (w - w_crop) // 2
	image = image[h_start:h_start + h_crop, w_start:w_start + w_crop]
	image = cv2.resize(image, (h,w), interpolation=cv2.INTER_AREA)
	return image



def compute_l2_norm_large_array(head_pose, limit_head_pose, chunk_size=5000):
    indices = []
    radian_threshold = limit_head_pose * np.pi / 180  # Convert limit to radians
    
    # Process in chunks
    for start in tqdm(range(0, head_pose.shape[0], chunk_size)):
        end = min(start + chunk_size, head_pose.shape[0])
        head_pose_chunk = head_pose[start:end]
        
        # Compute L2 norm manually to avoid intermediate arrays
        norms = np.sqrt(np.sum(head_pose_chunk ** 2, axis=1))
        
        # Get indices that meet the condition
        chunk_indices = np.where(norms < radian_threshold)[0] + start
        indices.extend(chunk_indices)
    
    return np.array(indices)




def recover_image( image_tensor, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5]):
	"""
	read a tensor and recover it to image in cv2 format
	args:
		image_tensor: [C, H, W] or [B, C, H, W]
	return:
		image_save: [B, H, W, C]
	"""
	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(0)

	x = torch.mul(image_tensor, torch.FloatTensor(STD).view(3,1,1).to(image_tensor.device))
	x = torch.add(x, torch.FloatTensor(MEAN).view(3,1,1).to(image_tensor.device) )
	x = x.data.cpu().numpy()
	# [C, H, W] -> [H, W, C]
	image_rgb = np.transpose(x, (0, 2, 3, 1))
	# RGB -> BGR
	image_bgr = image_rgb[:, :, :, [2,1,0]]
	# float -> int
	image_save = np.clip(image_bgr*255, 0, 255).astype('uint8')

	return image_save
