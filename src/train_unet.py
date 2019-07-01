# import time
# import os
# import numpy as np
# from collections import defaultdict
# from average_meter import AverageMeter
# import logging
# import torch
#
# import utils
# from options import UnetConfiguaration, TrainingOptions
# from model.unet.unet_model import UnetModel
#
#
# def train_unet(model: UnetModel,
#                device: torch.device,
#                unet_config: UnetConfiguaration,
#                train_options: TrainingOptions,
#                this_run_folder: str):
#     train_data, val_data = utils.get_data_loaders(train_options)
#     file_count = len(train_data.dataset)
#     if file_count % train_options.batch_size == 0:
#         steps_in_epoch = file_count // train_options.batch_size
#     else:
#         steps_in_epoch = file_count // train_options.batch_size + 1
#
#     print_each = 10
#     images_to_save = 8
#     saved_images_size = (512, 512)
#
#     for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
#         logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
#         logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
#         training_losses = defaultdict(AverageMeter)
#         epoch_start = time.time()
#         step = 1
#         # pbar = tqdm(train_data)
#         for image, _ in train_data:
#         # for image, _ in pbar:
#             image = image.to(device)
#             message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], unet_config.message_length))).to(device)
#             losses, _ = model.train_on_batch(image, message)
#
#             for name, loss in losses.items():
#                 training_losses[name].update(loss)
#             if step % print_each == 0 or step == steps_in_epoch:
#                 logging.info(
#                     'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
#                 logging.info(utils.losses_to_string(training_losses))
#                 logging.info('-' * 40)
#                 # pbar.set_description(f'loss: {training_losses}')
#             # pbar.set_description('\n'+utils.losses_to_string(training_losses))
#             # pbar.set_postfix()
#             step += 1
#
#         train_duration = time.time() - epoch_start
#         logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
#         logging.info('-' * 40)
#         utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
#         # if tb_logger is not None:
#         #     tb_logger.save_losses(training_losses, epoch)
#         #     tb_logger.save_grads(epoch)
#         #     tb_logger.save_tensors(epoch)
#
#         first_iteration = True
#         validation_losses = defaultdict(AverageMeter)
#         logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
#         for image, _ in val_data:
#             image = image.to(device)
#             message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], unet_config.message_length))).to(device)
#             losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(image, message)
#             for name, loss in losses.items():
#                 validation_losses[name].update(loss)
#             if first_iteration:
#                 utils.save_images(image.cpu()[:images_to_save, :, :, :],
#                                   encoded_images[:images_to_save, :, :, :].cpu(),
#                                   epoch,
#                                   os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
#                 first_iteration = False
#
#         logging.info(utils.losses_to_string(training_losses))
#         logging.info('-' * 40)
#         utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
#         logging.info('we are here!!!!!!!!!!!!!!!!')
#         utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
#                            time.time() - epoch_start)
#
