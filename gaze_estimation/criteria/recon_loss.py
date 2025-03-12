
import lpips
import torch
import torch.nn as nn

# ----------------------- repo modules -------------------
from utils import instantiate_from_cfg

class LPIPSLoss(nn.Module):
    def __init__(self, device = None):
        super(LPIPSLoss, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.lpips_net = lpips.LPIPS(net='vgg')
        self.lpips_net.requires_grad = False
        self.lpips_net.eval()
        self.lpips_net.to(device)

    def forward(self, y_hat, y):
        loss = self.lpips_net(y_hat, y)
        return loss
    

class GazeHeadConsistencyLoss(nn.Module):
    def __init__(self, 
                subnet_gazehead_config, 
                loss_config):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = instantiate_from_cfg(subnet_gazehead_config)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad = False

        self.loss_fn = instantiate_from_cfg(loss_config)
    def forward(self, y_hat, y):
        pred_gaze_hat = self.model(y_hat)['pred_gaze']
        pred_gaze = self.model(y)['pred_gaze']
        return self.loss_fn(pred_gaze_hat, pred_gaze)
    