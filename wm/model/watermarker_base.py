import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from train.loss_names import LossNames
from noise.noiser import Noiser
from collections import OrderedDict


class WatermarkerBase():
    def __init__(self, **kwargs):
        super(WatermarkerBase, self).__init__()
        self.device = torch.device(kwargs['device'])
        self.noiser =  Noiser(kwargs['noise'])
        self.noiser.to(self.device)

        self.config = kwargs

        self.encoder_decoder = self.create_encoder_decoder()
        self.discriminator = self.create_discriminator()

        # TODO: add optimizer learning rate here
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(), lr=kwargs['adam_lr'])
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(), lr=kwargs['adam_lr'])

        self.image_loss = nn.MSELoss().to(self.device)
        self.message_loss = nn.MSELoss().to(self.device)

        self.cover_label = 1
        self.encoded_label = 0


    def create_encoder_decoder(self):
        raise NotImplementedError()

    def create_discriminator(self):
        raise NotImplementedError()

    def gan_loss(self, predicted, target_label):
        raise NotImplementedError()    


    def train_on_batch(self, images: torch.Tensor, messages: np.ndarray):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover = self.gan_loss(predicted=d_on_cover, target_label=d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            d_on_encoded = self.discriminator(encoded_images.detach())
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            d_loss_on_encoded = self.gan_loss(predicted=d_on_encoded, target_label=d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            # g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            g_loss_adv = self.gan_loss(predicted=d_on_encoded_for_enc, target_label=g_target_label_encoded)

            g_loss_enc = self.image_loss(encoded_images, images)

            g_loss_dec = self.message_loss(decoded_messages, messages)
            g_loss = self.config['adv_loss_weight'] * g_loss_adv + self.config['enc_loss_weight'] * g_loss_enc \
                     + self.config.get('dec_loss_weight', 1) * g_loss_dec

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = OrderedDict({
            LossNames.network_loss.value: g_loss.item(),
            LossNames.encoder_mse.value: g_loss_enc.item(),
            LossNames.decoder_mse.value: g_loss_dec.item(),
            LossNames.bitwise.value: bitwise_avg_err,
            LossNames.gen_adv_bce.value: g_loss_adv.item(),
            LossNames.discr_cov_bce.value: d_loss_on_cover.item(),
            LossNames.discr_enc_bce.value: d_loss_on_encoded.item(),
            LossNames.discr_avg_bce.value: (d_loss_on_cover.item() + d_loss_on_encoded.item())/2
        })

        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, images: torch.Tensor, messages: np.ndarray, tb_writer: SummaryWriter=None):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param images: validation images
        :param messages: validation messages
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """

        # # if TensorboardX logging is enabled, save some of the tensors.
        # if self.tb_logger is not None:
        #     self.tb_logger.add_tensor('encoder_out', self.encoder_decoder.encoder._modules['final_layer'].weight)
        #     self.tb_logger.add_tensor('decoder_out', self.encoder_decoder.decoder._modules['linear'].weight)
        #     self.tb_logger.add_tensor('discrim_out', self.discriminator._modules['linear'].weight)

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover = self.gan_loss(predicted=d_on_cover, target_label=d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.gan_loss(predicted=d_on_encoded, target_label=d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            # g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            g_loss_adv = self.gan_loss(predicted=d_on_encoded_for_enc, target_label=g_target_label_encoded)
            

            g_loss_enc = self.image_loss(encoded_images, images)
            g_loss_dec = self.message_loss(decoded_messages, messages)
            g_loss = self.config['adv_loss_weight'] * g_loss_adv + self.config['enc_loss_weight'] * g_loss_enc \
                     + self.config.get('dec_loss_weight', 1) * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = OrderedDict({
            LossNames.network_loss.value: g_loss.item(),
            LossNames.encoder_mse.value: g_loss_enc.item(),
            LossNames.decoder_mse.value: g_loss_dec.item(),
            LossNames.bitwise.value: bitwise_avg_err,
            LossNames.gen_adv_bce.value: g_loss_adv.item(),
            LossNames.discr_cov_bce.value: d_loss_on_cover.item(),
            LossNames.discr_enc_bce.value: d_loss_on_encoded.item(),
            LossNames.discr_avg_bce.value: (d_loss_on_cover.item() + d_loss_on_encoded.item())/2
        })
        return losses, (encoded_images, noised_images, decoded_messages)

    def __str__(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))