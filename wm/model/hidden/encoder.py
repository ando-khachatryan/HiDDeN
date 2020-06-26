import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
from util.common import expand_message

class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, message_length: int, inner_channels: int, num_blocks: int):


        super(Encoder, self).__init__()
        self.conv_channels = inner_channels
        self.num_blocks = num_blocks

        layers = [ConvBNRelu(3, self.conv_channels, name='cbnr-1')]

        for i in range(self.num_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels, name=f'cbnr-{i+1}')
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + message_length,
                                             self.conv_channels, name='cbnr-after-concat')

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

    def get_tensors_for_logging(self):
        # raise NotImplementedError()
        return self.final_layer.weight.data.copy()

    def get_grads_for_logging(self):
        return self.final_layer.weight.grad.copy()

