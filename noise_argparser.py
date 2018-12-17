import argparse
import re
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.noiser import Noiser


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return  (hmin, hmax), (wmin, wmax)



def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return {
        'type': 'crop',
        'height_ratios': (hmin, hmax),
        'width_ratios': (wmin, wmax)
    }


def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    heights = matches.groups()[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = matches.groups()[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])

    return {
        'type': 'cropout',
        'height_ratios': (hmin, hmax),
        'width_ratios': (wmin, wmax)
    }


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return {
        'type': 'dropout',
        'keep_ratio_range': (keep_min, keep_max)
    }


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )
    @staticmethod
    def parse_cropout_args(cropout_args):
        pass


    @staticmethod
    def parse_dropout_args(dropout_args):
        pass


    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                cropout_descriptor = parse_cropout(command)
                layers.append(cropout_descriptor)
            elif command[:len('crop')] == 'crop':
                crop_descriptor = parse_crop(command)
                layers.append(crop_descriptor)
            elif command[:len('dropout')] == 'dropout':
                dropout_descriptor = parse_dropout(command)
                layers.append(dropout_descriptor)
            elif command[:len('jpeg')] == 'jpeg':
                layers.append({
                    'type': 'jpeg_compression'
                })
            elif command[:len('identity')] == 'identity':
                layers.append({
                    'type': 'identity'
                })
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)



# class NoiseArgParser(argparse.Action):
#     def __init__(self,
#                  option_strings,
#                  dest,
#                  nargs=None,
#                  const=None,
#                  default=None,
#                  type=None,
#                  choices=None,
#                  required=False,
#                  help=None,
#                  metavar=None):
#         argparse.Action.__init__(self,
#                                  option_strings=option_strings,
#                                  dest=dest,
#                                  nargs=nargs,
#                                  const=const,
#                                  default=default,
#                                  type=type,
#                                  choices=choices,
#                                  required=required,
#                                  help=help,
#                                  metavar=metavar,
#                                  )
#         print('Initializing CustomAction')
#         for name, value in sorted(locals().items()):
#             if name == 'self' or value is None:
#                 continue
#             print('  {} = {!r}'.format(name, value))
#         print()
#         return

#     def __call__(self, parser, namespace, values,
#                  option_string=None):
#         print('Processing CustomAction for {}'.format(self.dest))
#         print('  parser = {}'.format(id(parser)))
#         print('  values = {!r}'.format(values))
#         print('  option_string = {!r}'.format(option_string))

#         # Do some arbitrary processing of the input values
#         if isinstance(values, list):
#             values = [v.upper() for v in values]
#         else:
#             values = values.upper()
#         # Save the results in the namespace using the destination
#         # variable given to our constructor.
#         setattr(namespace, self.dest, values)
#         print()