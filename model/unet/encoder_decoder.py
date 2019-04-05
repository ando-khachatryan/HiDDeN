import torch
import torch.nn as nn
from model.unet.encoder import UnetGenerator
from model.unet.decoder import RevealNet
from noise_layers.noiser import Noiser
from options import UnetConfiguaration


class UnetEncoderDecoder(nn.Module):
    def __init__(self, configuration: UnetConfiguaration, noiser: Noiser):
        super(UnetEncoderDecoder, self).__init__()
        self.encoder = UnetGenerator(input_nc=configuration.message_length+3, output_nc=3,
                                     num_downs=configuration.num_downs)
        self.noiser = noiser
        self.decoder = RevealNet()


    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        if self.noiser is not None:
            noised_image = self.noiser(encoded_image)
        else:
            noised_image = encoded_image

        decoded_wm = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_wm