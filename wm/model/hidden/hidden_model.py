import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from model.hidden.discriminator import Discriminator
from model.hidden.encoder_decoder import EncoderDecoder
from model.watermarker_base import WatermarkerBase
from noise.noiser import Noiser
from train.loss_names import LossNames


class Hidden(WatermarkerBase):
    def __init__(self, **kwargs):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__(**kwargs)
      
    
    def create_encoder_decoder(self):
        config = self.config
        return EncoderDecoder(message_length=config['message'], 
                encoder_channels=config['encoder_channels'], encoder_blocks=config['encoder_blocks'], 
                decoder_channels=config['decoder_channels'], decoder_blocks=config['decoder_blocks'], noiser=self.noiser).to(self.device)

    def create_discriminator(self):
        return Discriminator(inner_channels=self.config['discriminator_channels'], block_count=self.config['discriminator_blocks']).to(self.device)

