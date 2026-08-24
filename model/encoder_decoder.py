import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder -> Noiser -> Decoder.

    The individual encode/noise/decode stages are exposed so that
    the high-resolution residual watermark pipeline can be built
    around the existing 256x256 HiDDeN model.
    """

    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def encode(self, image, message):
        return self.encoder(image, message)

    def noise(self, encoded_image, cover_image):
        noised_and_cover = self.noiser([encoded_image, cover_image])
        return noised_and_cover[0]

    def decode(self, image):
        return self.decoder(image)

    def forward(self, image, message):
        encoded_image = self.encode(image, message)
        noised_image = self.noise(encoded_image, image)
        decoded_message = self.decode(noised_image)

        return encoded_image, noised_image, decoded_message