import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu


class RevealNet(nn.Module):
    def __init__(self, message_length, input_channels=3, decoder_inner_channels=64, decoder_inner_blocks=7):
        super(RevealNet, self).__init__()
        layers = [ConvBNRelu(input_channels, decoder_inner_channels)]

        for _ in range(decoder_inner_blocks):
            layers.append(ConvBNRelu(decoder_inner_channels, decoder_inner_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(decoder_inner_channels, message_length)


    def forward(self, input):
        decoded_message = self.model(input)
        # TODO: consider inplace
        decoded_message = decoded_message.squeeze(-1)
        decoded_message = decoded_message.squeeze(-1)
        decoded_message = self.linear(decoded_message)
        return decoded_message
