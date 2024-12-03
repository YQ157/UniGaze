import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw, angular_error

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
