
import torch
import numpy as np
import cv2
import torch.nn as nn



class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def worker_init_fn(_):
	worker_info = torch.utils.data.get_worker_info()
	dataset = worker_info.dataset
	worker_id = worker_info.id
	return np.random.seed(np.random.get_state()[1][0] + worker_id)



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



def align_images(image_list, h, w):
	if len(image_list) != h * w:
		# automatically calculate the number of rows, try to make it as square as possible
		h = int(np.sqrt(len(image_list)))
		w = int(np.ceil(len(image_list) / h))
		## if the number of images is not a perfect square, add blank images to the list
		image_list += [np.zeros_like(image_list[0])] * (h * w - len(image_list))
	
	rows = [image_list[i * w:(i + 1) * w] for i in range(h)]
	row_images = [cv2.hconcat(row) for row in rows]
	final_image = cv2.vconcat(row_images)
	return final_image



