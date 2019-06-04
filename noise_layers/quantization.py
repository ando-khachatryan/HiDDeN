import numpy as np
import torch
import torch.nn as nn


def transform(tensor: torch.Tensor, target_range):
    source_min = tensor.min()
    source_max = tensor.max()
    # normalize to [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # move to target range
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target


class FourierRound(nn.Module):
    def __init__(self, fourier_coeff_count: int):
        super(FourierRound, self).__init__()

        weights = torch.tensor([np.power(-1, n)/(np.pi * n) for n in range(1, fourier_coeff_count+1)])
        scales = torch.tensor([2 * np.pi * n for n in range(1, fourier_coeff_count+1)])
        for _ in range(4):
            weights.unsqueeze_(-1)
            scales.unsqueeze_(-1)

        self.register_buffer('weights', weights)
        self.register_buffer('scales', scales)

    def forward(self, tensor):
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        z = torch.sum(z, dim=0)
        return tensor + z


class Quantization(nn.Module):
    def __init__(self, fourier_coeff_count: int, target_range=(0, 255)):
        super(Quantization, self).__init__()
        self.fourier_round = FourierRound(fourier_coeff_count)
        self.target_range = target_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        original_range = (noised_image.min(), noised_image.max())
        z = transform(noised_image, self.target_range)
        z = self.fourier_round(z)
        z = transform(z, original_range)
        return z
