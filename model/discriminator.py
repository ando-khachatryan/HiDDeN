import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        X.squeeze_()
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X