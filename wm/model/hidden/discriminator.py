import torch.nn as nn
from wm.model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):

    # newrun_parser.add_argument('--discriminator-blocks', default=defaults['discriminator-blocks'], type=str, help='Number of blocks in the discriminator.')
    # newrun_parser.add_argument('--discriminator-channels', default=defaults['channels'], type=str, help='Number of channels in discriminator blocks.')
    # newrun_parser.add_argument('--discriminator-block-type', default=defaults['discriminator-block-type'], 

    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, inner_channels: int, block_count: int):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, inner_channels)]
        for _ in range(block_count - 1):
            layers.append(ConvBNRelu(inner_channels, inner_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(inner_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X