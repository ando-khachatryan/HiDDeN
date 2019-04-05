import numpy as np
import torch
import torch.nn as nn
from model.unet.encoder_decoder import UnetEncoderDecoder
from noise_layers.noiser import Noiser
from options import UnetConfiguaration


def repeat_message(message, mult):
    shape = message.shape
    message.unsqueeze_(1).unsqueeze_(1)
    message = message.expand(-1, mult, mult, -1, -1)
    message = message.permute(0, 1, 3, 2, 4)
    message = message.reshape(shape[0], mult*shape[1], mult*shape[2])
    return message

class UnetModel:
    def __init__(self, configuration: UnetConfiguaration, device: torch.device, noiser: Noiser):
        super(UnetModel, self).__init__()

        self.encoder_decoder = UnetEncoderDecoder(configuration, noiser).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss()

        # self.cover_label =

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]
        self.encoder_decoder.train()

        with torch.enable_grad():
            self.optimizer_enc_dec.zero_grad()
            encoded_images, decoded_messages = self.encoder_decoder(images, messages)
            enc_loss = self.mse_loss(images, encoded_images)
            dec_loss = self.mse_loss(messages, decoded_messages)
            loss = dec_loss + self.config.encoder_loss_coeff * dec_loss
            loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss:    ': loss.item(),
            'encoder  ': enc_loss.item(),
            'decoder  ': dec_loss.item(),
            'bitwise  ': bitwise_avg_err
        }
        return losses


    def validate_on_batch(self):
        pass


    def to_string(self):
        return str(self.encoder_decoder)
