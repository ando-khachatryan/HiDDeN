import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter


def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def main():
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    # parser.add_argument('--size', '-s', default=128, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the data is stored.')
    parser.add_argument('--runs_root', '-r', default=os.path.join('.', 'experiments'), type=str,
                        help='The root folder where data about experiments are stored.')
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='Validation batch size.')

    args = parser.parse_args()
    print_each = 25

    completed_runs = [o for o in os.listdir(args.runs_root)
                      if os.path.isdir(os.path.join(args.runs_root, o)) and o != 'no-noise-defaults']

    print(completed_runs)

    write_csv_header = True
    for run_name in completed_runs:
        current_run = os.path.join(args.runs_root, run_name)
        print(f'Run folder: {current_run}')
        options_file = os.path.join(current_run, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        train_options.train_folder = os.path.join(args.data_dir, 'val')
        train_options.validation_folder = os.path.join(args.data_dir, 'val')
        train_options.batch_size = args.batch_size
        checkpoint, chpt_file_name = utils.load_last_checkpoint(os.path.join(current_run, 'checkpoints'))
        print(f'Loaded checkpoint from file {chpt_file_name}')

        noiser = Noiser(noise_config)
        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, checkpoint)

        print('Model loaded successfully. Starting validation run...')
        _, val_data = utils.get_data_loaders(hidden_config, train_options)
        file_count = len(val_data.dataset)
        if file_count % train_options.batch_size == 0:
            steps_in_epoch = file_count // train_options.batch_size
        else:
            steps_in_epoch = file_count // train_options.batch_size + 1

        losses_accu = {}
        step = 0
        for image, _ in val_data:
            step += 1
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message],
                                                                                                set_eval_mode=True)
            if not losses_accu:  # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = AverageMeter()
            for name, loss in losses.items():
                losses_accu[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                print(f'Step {step}/{steps_in_epoch}')
                utils.print_progress(losses_accu)
                print('-' * 40)

        # utils.print_progress(losses_accu)
        write_validation_loss(os.path.join(args.runs_root, 'validation_run.csv'), losses_accu, run_name,
                              checkpoint['epoch'],
                              write_header=write_csv_header)
        write_csv_header = False

    # train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()