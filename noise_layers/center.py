import torch.nn as nn
import numpy as np

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

class CenterCrop(nn.Module):
    def __init__(self, height_ratio_range, width_ratio_range):
        super().__init__()

        self.height_ratio_min = height_ratio_range[0]
        self.height_ratio_max = height_ratio_range[1]

        self.width_ratio_min = width_ratio_range[0]
        self.width_ratio_max = width_ratio_range[1]

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]

        _, _, height, width = noised_image.shape

        crop_h = int(
            height * random_float(
                self.height_ratio_min,
                self.height_ratio_max
            )
        )

        crop_w = int(
            width * random_float(
                self.width_ratio_min,
                self.width_ratio_max
            )
        )

        # Always center
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2

        noised_and_cover[0] = noised_image[
            :,
            :,
            top:top + crop_h,
            left:left + crop_w
        ]

        return noised_and_cover