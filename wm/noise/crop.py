import torch.nn as nn
import numpy as np
from noise.func import get_random_rectangle_inside

class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is 
    a uniform random between crop_min and crop_max
    """
    def __init__(self, crop_min: float, crop_max: float):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.crop_min = crop_min
        self.crop_max = crop_max


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, 
                                            height_ratio_range=(self.crop_min, self.crop_max), 
                                            width_ratio_range=(self.crop_min, self.crop_max)
        )

        return noised_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()

    def __repr__(self):
        return f'crop({self.crop_min},{self.crop_max})'

