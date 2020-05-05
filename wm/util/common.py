import numpy as np
import os
import re
import csv
import time
import pickle
import logging
from pathlib import Path

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, processed_images, filename, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    processed_images = processed_images[:processed_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    processed_images = (processed_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        processed_images = F.interpolate(processed_images, size=resize_to)

    stacked_images = torch.cat([images, processed_images], dim=0)
    # filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def update_checkpoint(model, experiment_name: str, epoch: int, checkpoint_folder: str, checkpoint_type: str):
    """Updates a checkpoint, or creates a new one if none exists"""
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if checkpoint_type not in ['last', 'best']:
        raise ValueError(
            f'Only supported checkpoint types are "last" and "best", but checkpoint_type was "{checkpoint_type}"')

    checkpoint_filename = f'{experiment_name}--{checkpoint_type}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    if not os.path.isfile(checkpoint_filename):
        logging.info(f'Saving {checkpoint_type }checkpoint to {checkpoint_filename}')
    else:
        logging.info(f'Overwriting {checkpoint_type} checkpoint file: {checkpoint_filename}, epoch: {epoch}')
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Checkpoint save/update done.')


def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(model, checkpoint):
    """ Restores the network object from a checkpoint object """
    model.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    model.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    model.discriminator.load_state_dict(checkpoint['discrim-model'])
    model.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def get_data_loaders(image_height, image_width, train_folder, validation_folder, batch_size):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((image_height, image_width), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=batch_size, 
                                                shuffle=False, num_workers=4)

    return train_loader, validation_loader


def losses_to_string(losses):
    max_len = max([len(loss_name) for loss_name in losses])
    log_strings = []
    for loss_name, loss_value in losses.items():
        log_strings.append(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))
    return '\n'.join(log_strings)


def get_timestamp():
    return time.strftime("%Y.%m.%d--%H-%M-%S")


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


def expand_message(message, spatial_height, spatial_width):
    expanded_message = message.unsqueeze(-1)
    expanded_message.unsqueeze_(-1)
    return expanded_message.expand(-1, -1, spatial_height, spatial_width)


