import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from model.unet.encoder_decoder import UnetEncoderDecoder
from model.hidden.discriminator import Discriminator
from model.watermarker_base import WatermarkerBase
from noise.noiser import Noiser
from train.loss_names import LossNames


class UnetModel(WatermarkerBase):
    def __init__(self, **kwargs):
        super(UnetModel, self).__init__(**kwargs)
    

    def create_encoder_decoder(self):
        config = self.config
        return UnetEncoderDecoder(network_variant=config['main_command'], noiser=self.noiser, message_length=config['message'],
                encoder_down_blocks=config['encoder_blocks'],
                decoder_blocks=config['decoder_blocks'],
                decoder_inner_channels=config['decoder_channels'],
                decoder_block_type=config['decoder_block_type'], 
                use_dropout=config['use_dropout']).to(self.device)  


    def create_discriminator(self):
        return Discriminator(inner_channels=self.config['discriminator_channels'], 
                    block_count=self.config['discriminator_blocks']).to(self.device)

