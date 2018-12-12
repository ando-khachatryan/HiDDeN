import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils

from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser



def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """


    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        print('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        print('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        losses_accu = {}
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses = model.train_on_batch([image, message])
            if not losses_accu: # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = []

            for name, loss in losses.items():
                losses_accu[name].append(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                print('Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.print_progress(losses_accu)
                print('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        print('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        print('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), losses_accu, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(losses_accu, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        print('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses = model.validate_on_batch([image, message])
            if not losses_accu: # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = []
            for name, loss in losses.items():
                losses_accu[name].append(loss)
            step += 1
        utils.print_progress(losses_accu)
        print('-' * 40)
        utils.save_checkpoint(model, epoch, losses_accu, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), losses_accu, epoch, time.time() - epoch_start)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    parser.add_argument('--size', '-s', default=128, type=int)
    parser.add_argument('--data-dir', '-d', required=True, type=str)

    parser.add_argument('--runs-folder', '-sf', default=os.path.join('.', 'runs'), type=str)
    parser.add_argument('--message', '-m', default=30, type=int)
    parser.add_argument('--epochs', '-e', default=400, type=int)
    parser.add_argument('--batch-size', '-b', required=True, type=int)
    parser.add_argument('--continue-from-folder', '-c', default='', type=str)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')
    parser.set_defaults(tensorboard=True)

    args = parser.parse_args()

    checkpoint = None
    if args.continue_from_folder != '':
        this_run_folder = args.continue_from_folder
        train_options, hidden_config, noise_config = utils.load_options(this_run_folder)
        checkpoint = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch)

        # noise_config = [
        #     {
        #         'type': 'resize',
        #         'resize_ratio': 0.4
        # }]
        noise_config = []
        hidden_config = HiDDenConfiguration(H=args.size, W=args.size,
                                            message_length=args.message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3
                                            )

        this_run_folder = utils.create_folder_for_run(train_options)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)

    noiser = Noiser(noise_config, device)

    if args.tensorboard:
        print('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None

    model = Hidden(hidden_config, device, noiser, tb_logger)

    if args.continue_from_folder != '':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        utils.model_from_checkpoint(model, checkpoint)

    print('HiDDeN model: {}\n'.format(model.to_stirng()))
    print('Model Configuration:\n')
    pprint.pprint(vars(hidden_config))
    print('\nNoise configuration:\n')
    pprint.pprint(str(noise_config))
    print('\nTraining train_options:\n')
    pprint.pprint(vars(train_options))
    print()

    train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()
