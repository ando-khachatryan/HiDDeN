import numpy as np
import re
import torch.nn as nn

# from noise import JpegCompression, Quantization, GaussianBlur, Crop, Cropout, Dropout, Resize, Identity
import noise
# from wm.noise.identity import Identity
# from wm.noise.crop import Crop
# from wm.noise.cropout import Cropout
# from wm.noise.dropout import Dropout
# from wm.noise.gaussian_blur import GaussianBlur
# from wm.noise.quantization import Quantization
# from wm.noise.resize import Resize
# from wm.noise.jpeg_compression import JpegCompression


class Noiser(nn.Module):
    @staticmethod 
    def layers_from_string(noise_command):
        layers = []
        if noise_command.strip() == '':
            return layers
        split_commands = noise_command.split('+')
        for command in split_commands:
            command = command.replace(' ', '')
            if command == 'jpeg()':
                layers.append(noise.JpegCompression())
            elif command == 'quant()':
                layers.append(noise.Quantization())
            elif command.startswith('blur'):
                matches = re.match(r'(blur)\((\d+(\.\d+)?)?\)', command)
                match_groups = matches.groups()
                if match_groups[1]:
                    layers.append(noise.GaussianBlur(sigma=float(match_groups[1])))
                else:
                    layers.append(noise.GaussianBlur())
            else:
                matches = re.match(r'(cropout|crop|resize|dropout)\((\d+\.*\d*,\d+\.*\d*)\)', command)
                match_groups = matches.groups()
                layer_name = match_groups[0]
                args_range = match_groups[1].split(',')
                lbound, ubound = float(args_range[0]), float(args_range[1])
                if lbound < 0:
                    raise ValueError(f'The lower bound should be positive, but it is: {lbound}')
                if lbound > ubound:
                    raise ValueError(f'Error parsing command={command}. The first argument not be larger than the second. '  
                                     f'First={lbound}, Second={ubound}')
                if layer_name == 'cropout':
                    layers.append(noise.Cropout(lbound, ubound))
                elif layer_name == 'crop':
                    layers.append(noise.Crop(lbound, ubound))
                elif layer_name == 'dropout':
                    layers.append(noise.Dropout(lbound, ubound))
                elif layer_name == 'resize':
                    layers.append(noise.Resize(lbound, ubound))
                else:
                    raise ValueError (f'Layer {layer_name} not supported. Full command = {command}')
        
        return layers

    def __init__(self, noise_command: str):
        super(Noiser, self).__init__()
        noise_layers = Noiser.layers_from_string(noise_command=noise_command)
        self.noise_layers = [noise.Identity()] + noise_layers

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

    def to(self, device):
        super(Noiser, self).to(device)
        layers_on_device = []
        for layer in self.noise_layers:
            layers_on_device.append(layer.to(device))
        self.noise_layers = layers_on_device

    def __repr__(self):
        layers_str = [str(layer) for layer in self.noise_layers]
        layers_str.sort()
        return '+'.join(layers_str)
