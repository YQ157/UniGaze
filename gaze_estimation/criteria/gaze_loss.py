from collections import OrderedDict
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


from utils.gaze_transform import rotation_matrix_2d, rotate_pitchyaw

from gazelib.draw.draw_image import draw_gaze
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw


# from models.resnet import resnet50, resnet18


def cosine_similarity_loss(embedding1, embedding2):
	# cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
	# cos_sim = F.hardtanh(cos_sim, -1.0, 1.0)
	# cos_sim = torch.mean(cos_sim)
	# return 1 - cos_sim  # Aim to minimize difference (maximize similarity)
	cos_critertion = nn.CosineSimilarity(dim=1, eps=1e-6)
	cosine_sim = cos_critertion(embedding1, embedding2)
	loss = 1 - cosine_sim
	loss = torch.mean(loss)
	return loss

def nn_angular_distance(a, b):
	sim = F.cosine_similarity(a, b, eps=1e-6)
	sim = F.hardtanh(sim, -1.0, 1.0)
	return torch.acos(sim) * (180 / np.pi)


def gaze_angular_loss( y, y_hat):
	y = pitchyaw_to_vector(y)
	y_hat = pitchyaw_to_vector(y_hat)
	loss = nn_angular_distance(y, y_hat)
	return torch.mean(loss)


class AngularLoss(nn.Module):
	def __init__(self, epsilon=None):
		super().__init__()
		self.epsilon = epsilon

	def forward(self, y, y_hat):
		y = pitchyaw_to_vector(y)
		y_hat = pitchyaw_to_vector(y_hat)
		loss = nn_angular_distance(y, y_hat)
		if self.epsilon is not None:
			loss = torch.where(loss < self.epsilon, torch.zeros_like(loss), loss - self.epsilon)
		return torch.mean(loss)
	  
	def forward(self, y, y_hat, weight=None, average=True):
		y = pitchyaw_to_vector(y)
		y_hat = pitchyaw_to_vector(y_hat)
		loss = nn_angular_distance(y, y_hat)
		if self.epsilon is not None:
			loss = torch.where(loss < self.epsilon, torch.zeros_like(loss), loss - self.epsilon)
		if weight is not None:
			loss = loss * weight
		# return torch.mean(loss)

		non_zero_loss = loss[loss > 0]
		number_of_non_zero_loss = non_zero_loss.numel()
		
		if number_of_non_zero_loss > 0:  # Check if there are any non-zero loss elements
			if average:
				return torch.sum( non_zero_loss, dim=0) / number_of_non_zero_loss
			else:
				return non_zero_loss
		else: # Return 0 if all losses are zero
			if average:
				return torch.tensor(0.0, device=loss.device)
			else:
				return loss
	
	

class PitchYawLoss(nn.Module):
	def __init__(self, loss_type='l1', epsilon=None):
		super().__init__()
		self.loss_type = loss_type
		self.epsilon = epsilon


	def gaze_l2_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) **2   
		loss = torch.mean(loss, dim=1) 
		if self.epsilon is not None:
			loss = torch.where(loss < self.epsilon, torch.zeros_like(loss), loss - self.epsilon)

		return loss 
		
	def gaze_l1_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) 
		loss = torch.mean(loss, dim=1) 
		if self.epsilon is not None:
			loss = torch.where(loss < self.epsilon, torch.zeros_like(loss), loss - self.epsilon)
		return loss

	def forward(self, y, y_hat, weight=None, average=True):
		if self.loss_type == 'l1':
			loss_all =  self.gaze_l1_loss(y, y_hat)
		elif self.loss_type == 'l2':
			loss_all = self.gaze_l2_loss(y, y_hat)
		else:
			raise NotImplementedError
		
		if weight is not None:
			loss_all = loss_all * weight

		non_zero_loss = loss_all[loss_all > 0]
		number_of_non_zero_loss = non_zero_loss.numel()

		if number_of_non_zero_loss > 0:  # Check if there are any non-zero loss elements
			if average:
				return torch.sum( non_zero_loss, dim=0) / number_of_non_zero_loss
			else:
				return non_zero_loss
		else:
			if average:
				return torch.tensor(0.0, device=loss_all.device)
			else:
				return loss_all




	

class LabelDifference(nn.Module):
	def __init__(self, distance_type='l1'):
		super().__init__()
		self.distance_type = distance_type

	def forward(self, labels):
		# labels: [bs, label_dim]
		# output: [bs, bs]
		if self.distance_type == 'l1':
			return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
		
		elif self.distance_type == 'l2':
			return (labels[:, None, :] - labels[None, :, :]).norm(2, dim=-1)
		
		elif self.distance_type == 'angular':
			# label_dim: [bs, 2] (assuming 2D vectors)
			# output: [bs, bs]
			bs = labels.size(0)  # Get the batch size
			angular_diff = torch.zeros(bs, bs)  # Initialize output tensor
			labels_vector = pitchyaw_to_vector(labels)

			cos_sim = torch.matmul(labels_vector, labels_vector.T)
			# Angular difference in radians
			cos_sim = F.hardtanh(cos_sim, -1.0, 1.0)
			angular_diff = torch.acos(cos_sim) * (180 / np.pi)
			return angular_diff
		
		else:
			raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
	def __init__(self, similarity_type='l2'):
		super(FeatureSimilarity, self).__init__()
		self.similarity_type = similarity_type

	def forward(self, features):
		# labels: [bs, feat_dim]
		# output: [bs, bs]
		if self.similarity_type == 'l1':
			return - torch.abs(features[:, None, :] - features[None, :, :]).sum(dim=-1)
		
		elif self.similarity_type == 'l2':
			return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
		
		elif self.similarity_type == 'angular':
			# x: [bs, 2] (assuming 2D vectors)
			# output: [bs, bs]
			features = F.normalize(features, dim=1)
			# Cosine similarity (represents the cosine of the angle between vectors)
			cos_sim = torch.matmul(features, features.T)
			# Angular difference in radians
			cos_sim = F.hardtanh(cos_sim, -1.0, 1.0)
			angular_diff = torch.acos(cos_sim) * (180 / np.pi)
			return 180 - angular_diff
		
		else:
			raise ValueError(self.distance_type)


class RnCLoss_degrade(nn.Module):
	def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
		super().__init__()
		self.t = temperature
		self.label_diff_fn = LabelDifference(label_diff)
		self.feature_sim_fn = FeatureSimilarity(feature_sim)

	# def forward(self, features, labels):
	#     # features: [bs, 2, feat_dim]
	#     # labels: [bs, label_dim]
	#     features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
		
	# def forward(self, features, labels):
	#     # features: [ 2 * bs, feat_dim]
	#     # labels: [  bs, label_dim]
	#     labels = labels.repeat(2, 1)  # [2bs, label_dim]

	#     label_diffs = self.label_diff_fn(labels)
	#     logits = self.feature_sim_fn(features).div(self.t)
	#     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
	#     logits -= logits_max.detach()
	#     exp_logits = logits.exp()

	#     n = logits.shape[0]  # n = 2bs

	#     # remove diagonal
	#     logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
	#     exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
	#     label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

	#     loss = 0.
	#     for k in range(n - 1):
	#         pos_logits = logits[:, k]  # 2bs
	#         pos_label_diffs = label_diffs[:, k]  # 2bs
	#         neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
	#         pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
	#         loss += - (pos_log_probs / (n * (n - 1))).sum()

	#     return loss
	


	def forward(self, features, labels):
		# # features: [bs, 2, feat_dim]
		# # labels: [bs, label_dim]
		# features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
		## features: [2 * bs, feat_dim]

		labels = labels.repeat(2, 1)  # [2bs, label_dim]
		label_diffs = self.label_diff_fn(labels)
		logits = self.feature_sim_fn(features).div(self.t)

		n = logits.shape[0]  # n = 2bs
		
		# remove diagonal ## NOTE: moved earlier
		logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
		
		logits_max, _ = torch.max(logits, dim=1, keepdim=True)
		logits -= logits_max.detach()
		exp_logits = logits.exp()

		# remove diagonal
		# logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
		# exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
		label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

		loss = 0.
		for k in range(n - 1):
			pos_logits = logits[:, k]  # 2bs
			pos_label_diffs = label_diffs[:, k]  # 2bs
			neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
			pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
			loss += - (pos_log_probs / (n * (n - 1))).sum()
		return loss