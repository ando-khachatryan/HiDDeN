import time
import os
import numpy as np
from collections import defaultdict
from average_meter import AverageMeter
import logging
import torch
# from tqdm import tqdm

from src import utils
from options import TrainingOptions
from model.hidden.hidden import Hidden
from model.unet.unet_model import UnetModel
from loss_names import LossNames


def train(model,
          device: torch.device,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains a watermark embedding-extracting model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    train_data, val_data = utils.get_data_loaders(train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    best_validation_error = np.inf

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], model.config.message_length))).to(device)
            losses, _ = model.train_on_batch(image, message)

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                logging.info(utils.losses_to_string(training_losses))
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)

        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], model.config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(image, message)
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                # if model.config.enable_fp16:
                #     image = image.float()
                #     encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        logging.info(utils.losses_to_string(validation_losses))
        logging.info('-' * 40)
        utils.update_checkpoint(model, train_options.experiment_name, epoch,
                                os.path.join(this_run_folder, 'checkpoints'), 'last')

        if isinstance(model, Hidden):
            network_loss = validation_losses[LossNames.hidden_loss.value].avg
        elif isinstance(model, UnetModel):
            network_loss = validation_losses[LossNames.unet_loss.value].avg
        else:
            raise ValueError('Only "hidden" or "unet" networks are supported')

        if network_loss < best_validation_error:
            utils.update_checkpoint(model, train_options.experiment_name, epoch,
                                    os.path.join(this_run_folder, 'checkpoints'), 'best')
            best_validation_error = network_loss

        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
