import torch
import torch.nn as nn
from src.options import HiDDenConfiguration
from src.model.conv_bn_relu import ConvBNRelu
from src.utils import expand_message

class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = expand_message(message, spatial_height=image.shape[2], spatial_width=image.shape[3])

        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
