# encoding: utf-8

"""
Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

"""
import functools
import torch
import torch.nn as nn

from utils import expand_message


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

class UnetGenerator(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, num_downs: int, message_length: int,
                 ngf: int = 64, output_function=nn.Tanh):
        # def __init__(self, input_nc: int, output_nc: int, num_downs: int, message_length: int,
        #              ngf: int = 64, norm_layer=nn.BatchNorm2d, use_dropout: bool = False, output_function=nn.Sigmoid):
        super(UnetGenerator, self).__init__()

        norm_layer = nn.BatchNorm2d
        use_dropout = False

        unet_block = UnetSkipConnectionBlock(outer_nc=ngf * 8, inner_nc=ngf * 8, input_nc=ngf * 8 + message_length,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(outer_nc=ngf * 8, inner_nc=ngf * 8, input_nc=ngf * 8 + message_length,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(outer_nc=ngf * 4, inner_nc=ngf * 8, input_nc=ngf * 4 + message_length,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(outer_nc=ngf * 2, inner_nc=ngf * 4, input_nc=ngf * 2 + message_length,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(outer_nc=ngf, inner_nc=ngf * 2, input_nc=ngf + message_length,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(outer_nc=output_nc, inner_nc=ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

    def forward(self, image, message):
        return self.model(image, message)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up.append(nn.Dropout(0.5))

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)
        # self.model = nn.Sequential(*model)

    def forward(self, x, message):
        expanded_message = expand_message(message, x.shape[2], x.shape[3])
        x_cat = torch.cat((x, expanded_message), 1)
        image_down = self.down(x_cat)
        if self.submodule is not None: # means we are at the innermost module
            image_down = self.submodule(image_down, message)
        image_up = self.up(image_down)
        if not self.outermost:
            image_up = torch.cat([image_up, x], 1)
        return image_up

        # else:
        #     model_out = self.model(x, message)
        #     expanded_message = expand_message(message, x.shape[2], x.shape[3])
        #     return torch.cat((x, model_out, expanded_message), 1)

    # def forward(self, x):
    #     if self.outermost:
    #         return self.model(x)
    #     else:
    #         return torch.cat([x, self.model(x)], 1)
