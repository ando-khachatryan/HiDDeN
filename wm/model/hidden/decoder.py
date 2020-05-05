from wm.model.conv_bn_relu import ConvBNRelu
import torch.nn as nn

class Decoder(nn.Module):

    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, message_length: int, inner_channels: int, num_blocks: int):
        super(Decoder, self).__init__()

        layers = [ConvBNRelu(3, inner_channels)]
        for _ in range(num_blocks - 2):
            layers.append(ConvBNRelu(inner_channels, inner_channels))

        layers.append(ConvBNRelu(inner_channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
