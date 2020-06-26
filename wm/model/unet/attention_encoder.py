# encoding: utf-8

"""
Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

"""
import functools
import torch
import torch.nn as nn
from util.common import expand_message


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, message_length: int, mode: str='deep'):
        super(AttentionBlock, self).__init__()
        if mode not in ['deep', 'shallow']:
            raise ValueError(f'mode parameter should be either "deep" or "shallow", but "{mode}" was passed')
        
        self.mode = mode
        if mode == 'deep':
            attention_out_channels = message_length
        else:
            attention_out_channels = 1
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=attention_out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.merge_conv = nn.Conv2d(in_channels= message_length + in_channels, out_channels=out_channels, kernel_size=1)
        
    def forward(self, features, message):
        height, width = features.shape[2], features.shape[3]
        expanded_message = expand_message(message, spatial_height=height, spatial_width=width).clone()
        attn_coefs = self.attn(features)

        attended_message = torch.mul(attn_coefs, expanded_message)
        x = torch.cat([features, attended_message], dim=1)
        return self.merge_conv(x)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

class UnetAttnGenerator(nn.Module):

    _kwargs_defaults = {
        'unet_ngf': 64, 
        'unet_output_function': nn.Tanh,
        'unet_norm_layer': nn.BatchNorm2d,
        'use_dropout': False,
        'unet_down_blocks': 7
    }

    def __init__(self, num_downs: int, message_length: int, **kwargs):
        super(UnetAttnGenerator, self).__init__()
        ngf = kwargs.pop('unet_ngf', self._kwargs_defaults['unet_ngf'])
        output_function = kwargs.pop('unet_output_function', self._kwargs_defaults['unet_output_function'])
        norm_layer = kwargs.pop('unet_norm_layer', self._kwargs_defaults['unet_norm_layer'])
        use_dropout = kwargs.pop('use_dropout', self._kwargs_defaults['use_dropout'])
        num_downs = kwargs.pop('unet_down_blocks', self._kwargs_defaults['unet_down_blocks'])

        unet_block = UnetEmbeddingBlock(input_nc=ngf * 8, inner_nc=ngf * 8, message_length=message_length,
                                             module_name='innermost',
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetEmbeddingBlock(input_nc=ngf * 8, inner_nc=ngf * 8, 
                                                module_name=f'module-{i+1}',
                                                message_length=message_length,
                                                submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetEmbeddingBlock(input_nc=ngf * 4, inner_nc=ngf * 8, 
                                            module_name=f'module-{num_downs-3}',
                                            message_length=message_length,
                                            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetEmbeddingBlock(input_nc=ngf * 2, inner_nc=ngf * 4, 
                                            module_name=f'module-{num_downs-2}', 
                                            message_length=message_length,
                                            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetEmbeddingBlock(input_nc=ngf, inner_nc=ngf * 2, module_name=f'module-{num_downs-1}', 
                                            message_length=message_length,
                                            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetEmbeddingBlock(input_nc=3, inner_nc=ngf, outer_nc=3, module_name=f'outermost', 
                                            message_length=message_length, 
                                            submodule=unet_block, outermost=True, 
                                            norm_layer=norm_layer, use_dropout=use_dropout)

        self.model = unet_block

    def forward(self, image, message):
        return self.model(image, message)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetEmbeddingBlock(nn.Module):
    def __init__(self, input_nc, inner_nc, module_name, message_length=0, outer_nc=None,  
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetEmbeddingBlock, self).__init__()

        self.module_name = module_name
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if outer_nc is None:
            outer_nc = input_nc
        # if use_attention_block:
        #     self.attention_block = AttentionBlock(in_channels=input_nc, out_channels=outer_nc, messgae_len)
        
        if message_length == 0:
            raise ValueError(f'Message_length must be > 0')
        if not outermost:
            self.embed_message = AttentionBlock(in_channels=input_nc, out_channels=outer_nc, message_length=message_length)
        downconv_in_channels = input_nc    
        
        downconv = nn.Conv2d(in_channels=downconv_in_channels, 
                            out_channels=inner_nc, kernel_size=4,
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
            up = [uprelu, upconv, nn.Tanh()]
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
        image_down = self.down(x)    

        if self.submodule is not None: 
            # means we are not at the innermost module
            image_down = self.submodule(image_down, message)

        image_up = self.up(image_down)

        if not self.outermost:
            x = self.embed_message(x, message)
            # we are not the outermost module
            image_up = torch.cat([image_up, x], 1)
        return image_up


