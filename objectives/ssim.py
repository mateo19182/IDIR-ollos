import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import matplotlib.pyplot as plt

class SSIM_(_Loss):

    def __init__(self):
        super().__init__()
        self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        # cant use ms_ssim because it requires bigger images, have torch.Size(10000)
        #size = int(np.sqrt(fixed.shape[0]))
        # size=100
        # fixed = fixed.reshape(1, 1, size, size)
        # warped = warped.reshape(1, 1, size, size)
        #sim_val = ms_ssim(fixed, warped, data_range=1, size_average=True)
        ssim_val = ssim(fixed, warped, data_range=1, size_average=True)

        # plt.figure()
        # plt.imshow(fixed.detach().cpu()[0, 0, :, :], cmap='gray')
        # plt.title('Fixed Image')

        # plt.figure()
        # plt.imshow(warped.detach().cpu()[0, 0, :, :], cmap='gray')
        # plt.title('Warped Image')

        # plt.show()

        return -ssim_val