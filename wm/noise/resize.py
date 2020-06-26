import torch.nn as nn
import torch.nn.functional as F
from noise.func import random_float

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_min, resize_ratio_max, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_min
        self.resize_ratio_max = resize_ratio_max
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        return noised_image

    def __repr__(self):
        return f'resize({self.resize_ratio_min},{self.resize_ratio_max})'
