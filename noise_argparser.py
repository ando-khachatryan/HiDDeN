import argparse
import re
import torch
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.center import CenterCrop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.color import ColorJitter, ColorGrading, SharpnessEnhance


def parse_range(range_str: str) -> tuple[float, float]:
    clean = range_str.strip('()')
    low, high = map(float, clean.split(','))
    return (low, high)


def parse_crop(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            parts = re.findall(r'\([^)]+\)', inner)
            if len(parts) == 2:
                return Crop(parse_range(parts[0]), parse_range(parts[1]))
            elif len(parts) == 1:
                r = parse_range(parts[0])
                return Crop(r, r)
        except Exception:
            pass
    # Your default
    return Crop((0.7, 0.9), (0.7, 0.9))


def parse_center_crop(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            parts = re.findall(r'\([^)]+\)', inner)
            if len(parts) == 2:
                return CenterCrop(parse_range(parts[0]), parse_range(parts[1]))
            elif len(parts) == 1:
                r = parse_range(parts[0])
                return CenterCrop(r, r)
        except Exception:
            pass
    # Your default
    return CenterCrop((0.7, 0.9), (0.7, 0.9))


def parse_cropout(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            parts = re.findall(r'\([^)]+\)', inner)
            if len(parts) == 2:
                return Cropout(parse_range(parts[0]), parse_range(parts[1]))
            elif len(parts) == 1:
                r = parse_range(parts[0])
                return Cropout(r, r)
        except Exception:
            pass
    # Your default
    return Cropout((0.7, 0.9), (0.7, 0.9))


def parse_dropout(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            return Dropout(parse_range(inner))
        except Exception:
            pass
    # Your default
    return Dropout((0.8, 0.95))


def parse_resize(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            return Resize(parse_range(inner))
        except Exception:
            pass
    # Your default
    return Resize((0.5, 0.9))


def parse_color_jitter(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            parts = re.findall(r'\([^)]+\)', inner)
            if len(parts) == 3:
                return ColorJitter(parse_range(parts[0]), parse_range(parts[1]), parse_range(parts[2]))
        except Exception:
            pass
    return ColorJitter(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), saturation_range=(0.7, 1.3))


def parse_color_grading(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            return ColorGrading(temp_range=parse_range(inner))
        except Exception:
            pass
    return ColorGrading(temp_range=(0.05, 0.15))


def parse_sharpness(command: str = ""):
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            return SharpnessEnhance(factor_range=parse_range(inner))
        except Exception:
            pass
    return SharpnessEnhance(factor_range=(1.5, 2.5))


def parse_jpeg(command: str = "", device="cuda"):
    # Default is keeping (25, 9, 9) coefficients
    if '(' in command and ')' in command:
        try:
            inner = command[command.index('(') + 1 : command.rindex(')')]
            weights = tuple(int(x.strip()) for x in inner.split(','))
            if len(weights) == 3:
                return JpegCompression(device=device, yuv_keep_weights=weights)
        except Exception:
            pass
    return JpegCompression(device=device, yuv_keep_weights=(25, 9, 9))


class NoiseArgParser(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(NoiseArgParser, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        layers = []
        raw_val = values[0] if isinstance(values, list) else values
        split_commands = raw_val.split('+')

        for command in split_commands:
            command = command.strip().replace(' ', '')
            if not command:
                continue

            if command.startswith('center_crop'):
                layers.append(parse_center_crop(command))
            elif command.startswith('cropout'):
                layers.append(parse_cropout(command))
            elif command.startswith('crop'):
                layers.append(parse_crop(command))
            elif command.startswith('dropout'):
                layers.append(parse_dropout(command))
            elif command.startswith('resize'):
                layers.append(parse_resize(command))
            elif command.startswith('color_jitter'):
                layers.append(parse_color_jitter(command))
            elif command.startswith('color_grading'):
                layers.append(parse_color_grading(command))
            elif command.startswith('sharpness'):
                layers.append(parse_sharpness(command))
            elif command.startswith('jpeg'):
                layers.append(JpegCompression(device=getattr(namespace, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')))
            elif command.startswith('quant'):
                layers.append(Quantization())
            elif command.startswith('identity'):
                layers.append(Identity())
            elif command.startswith('jpeg'):
                # Ensure device is passed (e.g., from args or cuda)
                device = getattr(namespace, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
                layers.append(parse_jpeg(command, device=device))
            else:
                raise ValueError(f'Command not recognized: \n{command}')

        setattr(namespace, self.dest, layers)