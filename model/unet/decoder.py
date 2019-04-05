
import torch.nn as nn


class RevealNet(nn.Module):
    def __init__(self, channels_in = 3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(in_channels=nhf, out_channels=1, kernel_size=1, stride=1, padding=0),
            output_function()
        )

    def forward(self, input):
        output=self.layers(input)
        return output
