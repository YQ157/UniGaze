import torch
import torch.nn as nn




class PitchYawLoss(nn.Module):
	def __init__(self, loss_type='l1', epsilon=None):
		super().__init__()
		self.loss_type = loss_type
		self.epsilon = epsilon


	def gaze_l2_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) **2   
		loss = torch.mean(loss, dim=1) 

		return loss 
		
	def gaze_l1_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) 
		loss = torch.mean(loss, dim=1) 
		return loss

	def forward(self, y, y_hat, weight=None, average=True):
		if self.loss_type == 'l1':
			loss_all =  self.gaze_l1_loss(y, y_hat)
		elif self.loss_type == 'l2':
			loss_all = self.gaze_l2_loss(y, y_hat)
		else:
			raise NotImplementedError
		
		return torch.mean(loss_all, dim=0)
		
