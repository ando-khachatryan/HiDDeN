# import torch
# import torch.nn as nn


# class DownsampleBlock(nn.Module):
#     def __init__(self, input_nc, output_nc, normalize=nn.BatchNorm2d):
#         super(DownsampleBlock, self).__init__()
        
#         layers = [nn.Conv2d(input_nc, output_nc, kernel_size=4,
#                              stride=2, padding=1, bias=True)]
#         layers.append(nn.LeakyReLU(0.2, True))
#         if normalize is not None:
#             layers.append(nn.BatchNorm2d(output_nc))
#         self.down = nn.Sequential(*layers)

#     def forward(self, x):
#         image_down = self.down(x)
#         return image_down


# class UnetDecoder(nn.Module):
#     def __init__(self, module_config: dict, message_len: int, max_inner_channels = 1024, channel_mult=2):
#         super(UnetDecoder, self).__init__()

#         self.max_inner_channels = max_inner_channels
#         self.config = module_config
#         self.channel_mult = channel_mult

#         layers = []
#         in_channels = 3
#         out_channels = module_config[Module.Channels.value]
#         # layers.append(DownsampleBlock(input_nc=3, output_nc=out_channels))
#         # in_channels = out_channels
#         for i in range(self.config[Module.BlockCount.value]-1):
#             layers.append(DownsampleBlock(input_nc=in_channels, output_nc=out_channels))
#             in_channels = out_channels
#             if out_channels < self.max_inner_channels:
#                 out_channels = min(self.max_inner_channels, out_channels * self.channel_mult)

#         layers.append(DownsampleBlock(input_nc=in_channels, output_nc=out_channels, normalize=None))
#         layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
#         self.layers = nn.Sequential(*layers)
#         self.linear = nn.Linear(in_features=in_channels, out_features=message_len)
        
#     def forward(self, image_with_wm):
#         out = self.layers(image_with_wm)
#         out.squeeze_(3).squeeze_(2)
#         out = self.linear(out)
#         return out
