import sys
import argparse
import logging
import os
import pprint
import torch
import utils
import time
import numpy as np

from average_meter import AverageMeter
from options import UnetConfiguaration, TrainingOptions
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from model.unet.unet_model import UnetModel


def train(model: UnetModel,
          device: torch.device,
          unet_config: UnetConfiguaration,
          train_options: TrainingOptions,
          this_run_folder: str):
    train_data, val_data = utils.get_data_loaders((unet_config.H, unet_config.W), train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        losses_accu = {}
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], unet_config.message_length))).to(device)
            losses, _ = model.train_on_batch([image, message])
            if not losses_accu:  # dict is empty, initialize
                for name in losses:
                    # losses_accu[name] = []
                    losses_accu[name] = AverageMeter()

            for name, loss in losses.items():
                losses_accu[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(losses_accu)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), losses_accu, epoch, train_duration)
        # if tb_logger is not None:
        #     tb_logger.save_losses(losses_accu, epoch)
        #     tb_logger.save_grads(epoch)
        #     tb_logger.save_tensors(epoch)

        first_iteration = True

        utils.log_progress(losses_accu)
        logging.info('-' * 40)
        # utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        # utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), losses_accu, epoch,
        #                    time.time() - epoch_start)
        for loss in losses_accu:
            print(f'{loss}      :{losses_accu[loss]}')


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of Unet-watermark nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', default='/data/coco/10K', type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', default=32, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')

    new_run_parser.add_argument('--size', '-s', default=256, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=64, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    args = parent_parser.parse_args()

    assert args.command == 'new'
    start_epoch = 1
    train_options = TrainingOptions(
        batch_size=args.batch_size,
        number_of_epochs=args.epochs,
        train_folder=os.path.join(args.data_dir, 'train'),
        validation_folder=os.path.join(args.data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=start_epoch,
        experiment_name=args.name)

    noise_config = args.noise if args.noise is not None else []
    unet_config = UnetConfiguaration(H=args.size, W=args.size, num_downs=8, message_length=args.message,
                                     encoder_loss_coff=1.0)

    this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
    # with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
    #     pickle.dump(train_options, f)
    #     pickle.dump(noise_config, f)
    #     pickle.dump(hidden_config, f)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    # if (args.command == 'new' and args.tensorboard) or \
    #         (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
    #     logging.info('Tensorboard is enabled. Creating logger.')
    #     from tensorboard_logger import TensorBoardLogger
    #     tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    # else:
    #     tb_logger = None

    noiser = Noiser(noise_config, device)
    model = UnetModel(unet_config, device, noiser)

    logging.info('UNet model: {}\n'.format(model.to_string()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(unet_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, unet_config, train_options, this_run_folder)


if __name__ == '__main__':
    main()
