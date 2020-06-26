import torch.nn as nn
from model.hidden.encoder import Encoder
from model.hidden.decoder import Decoder
from noise.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, message_length, encoder_channels: int, encoder_blocks: int, 
            decoder_blocks: int, decoder_channels: int, noiser: Noiser, **kwargs):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(message_length=message_length, inner_channels=encoder_channels, 
                                num_blocks=encoder_blocks, **kwargs)
        self.noiser = noiser
        self.decoder = Decoder(message_length=message_length, inner_channels=decoder_channels, 
                num_blocks=decoder_blocks, **kwargs)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noiser([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
