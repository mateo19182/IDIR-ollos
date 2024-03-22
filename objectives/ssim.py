import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SSIM_(_Loss):

    def __init__(self):
        super().__init__()
        self.forward = self.metric


    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        # cant use ms_ssim because it requires bigger images, have torch.Size(10000)
        #print(fixed.shape, warped.shape)
        fixed = fixed.reshape(1, 1, 100, 100) ######
        warped = warped.reshape(1, 1, 100, 100) ######
        #sim_val = ms_ssim(fixed, warped, data_range=1, size_average=True)
        ssim_val = ssim(fixed, warped, data_range=1, size_average=True)
        return -ssim_val