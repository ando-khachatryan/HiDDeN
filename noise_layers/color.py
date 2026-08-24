import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ColorJitter(nn.Module):
    """
    Differentiable Color Jitter (Brightness, Contrast, Saturation).
    Input tensor expected in range [0, 1] with shape (B, 3, H, W).
    """
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), saturation_range=(0.7, 1.3)):
        super(ColorJitter, self).__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def forward(self, noised_and_cover):
        # In HiDDeN, noise layers usually take a tuple (noised_image, cover_image)
        # or just noised_image. Adjust unpacking to match your pipeline.
        if isinstance(noised_and_cover, (tuple, list)):
            x, cover = noised_and_cover
        else:
            x, cover = noised_and_cover, None

        B, C, H, W = x.shape
        device = x.device

        # 1. Random Brightness: I_new = I * factor
        if self.brightness_range:
            b_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.brightness_range)
            x = x * b_factor

        # 2. Random Contrast: I_new = (I - mean) * factor + mean
        if self.contrast_range:
            c_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.contrast_range)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * c_factor + mean

        # 3. Random Saturation: blend with grayscale
        if self.saturation_range:
            s_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.saturation_range)
            # ITU-R 601-2 luma transform weights
            luma_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
            grayscale = (x * luma_weights).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            x = grayscale + (x - grayscale) * s_factor

        x = torch.clamp(x, 0.0, 1.0)
        return (x, cover) if cover is not None else x


class ColorGrading(nn.Module):
    """
    Differentiable Color Temperature/Grading (Warm/Cool shifts).
    """
    def __init__(self, temp_range=(0.05, 0.15)):
        super(ColorGrading, self).__init__()
        self.temp_range = temp_range

    def forward(self, noised_and_cover):
        if isinstance(noised_and_cover, (tuple, list)):
            x, cover = noised_and_cover
        else:
            x, cover = noised_and_cover, None

        B, C, H, W = x.shape
        device = x.device

        # Sample shift factor and random warm (1) vs cool (-1) direction
        shift = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.temp_range)
        direction = torch.randint(0, 2, (B, 1, 1, 1), device=device).float() * 2 - 1 # -1 or 1

        # Warm (+R, -B) / Cool (-R, +B)
        r_mod = 1.0 + shift * direction
        b_mod = 1.0 - shift * direction

        r = x[:, 0:1, :, :] * r_mod
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :] * b_mod

        x = torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)
        return (x, cover) if cover is not None else x


class SharpnessEnhance(nn.Module):
    """
    Differentiable Sharpness Enhancement via Laplacian high-pass blending.
    """
    def __init__(self, factor_range=(1.5, 2.5)):
        super(SharpnessEnhance, self).__init__()
        self.factor_range = factor_range
        kernel = torch.tensor([[0., -1., 0.],
                               [-1., 4., -1.],
                               [0., -1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, noised_and_cover):
        if isinstance(noised_and_cover, (tuple, list)):
            x, cover = noised_and_cover
        else:
            x, cover = noised_and_cover, None

        B, C, H, W = x.shape
        device = x.device

        # Ensure kernel matches tensor device even if parent module missed .to(device)
        kernel = self.laplacian_kernel.to(device=device, dtype=x.dtype)

        edges = F.conv2d(x, kernel, padding=1, groups=3)
        factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.factor_range)

        x = torch.clamp(x + (factor - 1.0) * edges * 0.25, 0.0, 1.0)
        return (x, cover) if cover is not None else x