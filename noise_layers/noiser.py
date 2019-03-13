import torch
import torch.nn as nn
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.identity import Identity
from noise_layers.quantization import Quantization


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_config: list, device: torch.device):
        super(Noiser, self).__init__()

        self.noise_config = noise_config
        noise_layers = []

        for noise_layer_config in noise_config:
            layer_type = noise_layer_config['type'].lower()
            if layer_type == 'jpeg_compression':
                # TODO: Add jpeg compression level as a config option
                noise_layers.append(JpegCompression(device))
            elif layer_type == 'crop':
                noise_layers.append(Crop(noise_layer_config['height_ratios'], noise_layer_config['width_ratios']))
            elif layer_type == 'cropout':
                noise_layers.append(Cropout(noise_layer_config['height_ratios'], noise_layer_config['width_ratios']))
            elif layer_type == 'dropout':
                noise_layers.append(Dropout(noise_layer_config['keep_ratio_range']))
            elif layer_type == 'resize':
                if 'interpolation_method' in noise_layer_config:
                    noise_layers.append(Resize(noise_layer_config['resize_ratio_range'],
                                               noise_layer_config['interpolation_method']))
                else:
                    noise_layers.append(Resize(noise_layer_config['resize_ratio_range']))
            elif layer_type == 'rotate':
                pass
            elif layer_type == 'identity':
                noise_layers.append(Identity())
            elif layer_type == 'quantization':
                noise_layers.append(Quantization(device))
            else:
                raise ValueError('Noise layer of {} not supported'.format(noise_layer_config['type']))

        self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        return self.noise_layers(encoded_and_cover)
