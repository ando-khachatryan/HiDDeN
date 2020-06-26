import torch.nn as nn
from model.unet.encoder import UnetGenerator
from model.unet.attention_encoder import UnetAttnGenerator
from model.hidden.decoder import Decoder

from noise.noiser import Noiser
from util.configuration import BlockType


class UnetEncoderDecoder(nn.Module):
    def __init__(self, network_variant: str, noiser: Noiser, message_length: int, encoder_down_blocks: int, 
                decoder_inner_channels: int, decoder_blocks: int, **kwargs):

        super(UnetEncoderDecoder, self).__init__()
        if network_variant == 'unet-conv':
            self.encoder = UnetGenerator(num_downs=encoder_down_blocks, message_length=message_length, **kwargs)
        elif network_variant == 'unet-attn':
            self.encoder = UnetAttnGenerator(num_downs=encoder_down_blocks, message_length=message_length, **kwargs)
    
        if network_variant == 'unet-conv' or network_variant == 'unet-attn':
            self.decoder = Decoder(message_length=message_length, inner_channels=decoder_inner_channels, num_blocks=decoder_blocks)
        else:
            raise ValueError(f'Network variant {network_variant} not supported')
        
        self.network_variant = network_variant
        self.noiser = noiser
        

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        if self.noiser is not None:
            noised_image = self.noiser([encoded_image, image])
        else:
            noised_image = encoded_image

        decoded_wm = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_wm

