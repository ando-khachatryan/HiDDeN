import argparse
from argparse import ArgumentParser
import os
import pprint
import pickle
import logging
import sys

import torch

from src.noise_argparser import NoiseArgParser
from src import utils
from src.options import *
from src.model.unet.unet_model import UnetModel, UnetConfiguaration
from src.model.hidden.hidden import Hidden, HiDDenConfiguration
from src.noise_layers.noiser import Noiser


def create_new_run_subparser(new_run_parser: ArgumentParser):
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')

    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--enc-loss-weight', default=0.7, required=False, type=float,
                                help='The weight of encoder loss in the overall loss function')
    new_run_parser.add_argument('--adv-loss-weight', default=1e-3, required=False, type=float,
                                help='The weight of the adversarial loss in the overall loss function')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")

    new_run_parser.set_defaults(tensorboard=False, enable_fp16=False)


def create_continue_run_subparser(continue_run_parser: ArgumentParser):
    continue_run_parser.add_argument('--folder', '-f', required=True, type=str,
                                     help='Continue from the last checkpoint in this folder.')
    continue_run_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                     help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_run_parser.add_argument('--epochs', '-e', required=False, type=int,
                                     help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')


def prepare_training(network_type: str):
    if network_type not in ['hidden', 'unet']:
        raise ValueError(f'Expected network_type to be either "hidden" or "unet", but got "f{network_type}" instead')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')

    subparsers = parent_parser.add_subparsers(dest='command', required=True, help='New simulation or continue existing')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    create_new_run_subparser(new_run_parser)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    create_continue_run_subparser(continue_parser)

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, network_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
            os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name,
            image_height=args.size,
            image_width=args.size)

        noise_config = args.noise if args.noise is not None else []

        if network_type == 'hidden':
            network_config = HiDDenConfiguration(message_length=args.message,
                                                 encoder_blocks=4, encoder_channels=64,
                                                 decoder_blocks=7, decoder_channels=64,
                                                 use_discriminator=True,
                                                 discriminator_blocks=3, discriminator_channels=64,
                                                 decoder_loss=1,
                                                 encoder_loss=args.enc_loss_weight,
                                                 adversarial_loss=args.adv_loss_weight,
                                                 enable_fp16=args.enable_fp16
                                                 )
        else:  # network_type == 'unet'
            network_config = UnetConfiguaration(encoder_num_downs=7,
                                                decoder_blocks=7,
                                                discriminator_blocks=3,
                                                message_length=args.message,
                                                decoder_loss_weight=1,
                                                encoder_loss_weight=args.enc_loss_weight,
                                                adversarial_loss_weight=args.adv_loss_weight
                                                )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(network_config, f)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from src.tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None

    noiser = Noiser(noise_config, device)
    if network_type == 'hidden':
        model = Hidden(network_config, device, noiser, tb_logger)
    else:
        model = UnetModel(network_config, device, noiser)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('HiDDeN model: {}\n'.format(str(model)))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(network_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    return model, device, network_config, train_options, this_run_folder, tb_logger
