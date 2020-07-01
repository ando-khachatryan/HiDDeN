import time
import os
import numpy as np
from collections import defaultdict

import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from model.watermarker_base import WatermarkerBase
from model.hidden.hidden_model import Hidden
from model.unet.unet_model import UnetModel
from train.loss_names import LossNames
from train.average_meter import AverageMeter
import util.common as common


def train(model: WatermarkerBase,
        #   device: torch.device,
          job_name: str,
          job_folder: str,
          image_size: int, 
          train_folder: str,
          validation_folder: str, 
          batch_size: int,
          number_of_epochs: int,
          message_length: int,
          start_epoch: int, 
          tb_writer: SummaryWriter=None, 
          checkpoint_folder: str=None):
    """
    Trains a watermark embedding-extracting model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
    :return:
    """

    train_data, val_data = common.get_data_loaders(image_height=image_size, image_width=image_size, 
                        train_folder=train_folder, validation_folder=validation_folder, batch_size=batch_size)

    file_count = len(train_data.dataset)
    if file_count % batch_size == 0:
        steps_in_epoch = file_count // batch_size
    else:
        steps_in_epoch = file_count // batch_size + 1

    image_save_epochs = 10
    images_to_save = 8
    saved_images_size = (512, 512)
    if checkpoint_folder is None:
        checkpoint_folder = os.path.join(job_folder, 'checkpoints')
    best_validation_error = np.inf
    logging_losses = [LossNames.network_loss.value, LossNames.encoder_mse.value, LossNames.bitwise.value, LossNames.discr_avg_bce.value]

    # from pprint import pprint
    # import json
    # print('Debug hook config:')
    # with open('/opt/ml/input/config/debughookconfig.json', 'r') as f:
    #     pprint(json.load(f))
    # print('-'*60)

    logging.info('Starting the training loop...')
    for epoch in range(start_epoch, number_of_epochs + 1):
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1    
        for image, _ in train_data:
            image = image.to(model.device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(model.device)
            losses_on_batch, _ = model.train_on_batch(image, message)

            for name, loss in losses_on_batch.items():
                training_losses[name].update(loss)
            logging.info(common.losses_to_string({loss_key: training_losses[loss_key] for loss_key in logging_losses}))
            logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        common.write_losses(os.path.join(job_folder, 'train.csv'), training_losses, epoch, train_duration)

        if tb_writer:
            for loss_name in logging_losses:
                pass
                # tb_writer.add_scalar(tag=f'train/{loss_name}', scalar_value=training_losses[loss_name].avg, global_step=epoch) # TODO: TB_DEBUG

            
        first_iteration = True
        validation_losses = defaultdict(AverageMeter)

        logging.info('Running validation for epoch {}/{}'.format(epoch, number_of_epochs))
        for image, _ in val_data:
            image = image.to(model.device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(model.device)
            losses_on_batch, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(image, message)
            for name, loss in losses_on_batch.items():
                validation_losses[name].update(loss)
            if first_iteration and epoch % image_save_epochs == 0:
                # if model.net_config.enable_fp16:
                #     image = image.float()
                #     encoded_images = encoded_images.float()
                cover_cpu = (image[:images_to_save, :, :, :].cpu() + 1)/2
                encoded_cpu = (encoded_images[:images_to_save, :, :, :].cpu() + 1)/2
                common.save_images(cover_images=cover_cpu,
                                  processed_images=encoded_cpu,
                                  filename=os.path.join(job_folder, 'images', f'epoch-{epoch}.png'), 
                                  resize_to=saved_images_size)
                if tb_writer:
                    common.save_to_tensorboard(cover_images=cover_cpu, encoded_images=encoded_cpu, tb_writer=tb_writer, global_step=epoch)
                first_iteration = False

        # logging.info(common.losses_to_string(validation_losses))
        logging.info(common.losses_to_string({loss_key: validation_losses[loss_key] for loss_key in logging_losses}))
        logging.info('-' * 40)
        common.update_checkpoint(model, job_name, epoch, checkpoint_folder, 'last')

        if validation_losses[LossNames.network_loss.value].avg < best_validation_error:
            common.update_checkpoint(model, job_name, epoch, checkpoint_folder, 'best')
            best_validation_error = validation_losses[LossNames.network_loss.value].avg

        common.write_losses(os.path.join(job_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
        if tb_writer:
            for loss_name in logging_losses:
                pass
                # tb_writer.add_scalar(tag=f'validation/{loss_name}', scalar_value=validation_losses[loss_name].avg, 
                # global_step=epoch) # TODO: TB_DEBUG
            # tb_writer.flush()

    # if tb_writer: # TODO: TB_DEBUG
    #     metrics = {}
    #     for key in validation_losses:
    #         metrics[f'validation--{key}'] = validation_losses[key].avg
    #     tb_writer.add_hparams(hparam_dict=None, metric_dict=metrics)
    #     tb_writer.flush()
    #     tb_writer.close()
            