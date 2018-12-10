import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio, interpolation_method = 'nearest'):
        super(Resize, self).__init__()
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        new_size = int(np.rint(noised_image.shape[2] * self.resize_ratio))
        noised_and_cover[0] = F.interpolate(noised_image,
                                     size=new_size,
                                     mode=self.interpolation_method)

        return noised_and_cover
