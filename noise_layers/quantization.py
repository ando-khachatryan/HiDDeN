import torch
import torch.nn as nn

class Quantization(nn.Module):
    """Transforms the image from float to uint8"""
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, images):
        return (images.add(1) * 175.5).round().clamp(0,255).add(-1)
