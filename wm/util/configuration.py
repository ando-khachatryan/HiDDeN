from enum import Enum

class BlockType(Enum):
    Conv = 'Conv'
    Unet = 'Unet'

hidden_defaults = {
    'enc-loss-weight': 0.7,
    'adv-loss-weight': 0.001,
    'encoder-blocks': 4,
    'encoder-block-type': BlockType.Conv.value,
    'decoder-blocks': 8,
    'decoder-block-type': BlockType.Conv.value,
    'discriminator-blocks': 3,
    'discriminator-block-type': BlockType.Conv.value,
    'channels': 64,
    'adam-lr': 1e-3
}

unet_conv_defaults = {
    'enc-loss-weight': 4,
    'adv-loss-weight': 0.001,
    'encoder-blocks': 7,
    'encoder-block-type': BlockType.Unet.value,
    'decoder-blocks': 8,
    'decoder-block-type': BlockType.Conv.value,
    'discriminator-blocks': 3,
    'discriminator-block-type': BlockType.Conv.value,
    'channels': 64,
    'adam-lr': 1e-3
}

unet_down_defaults = {
    'enc-loss-weight': 2,
    'adv-loss-weight': 0.001,
    'encoder-blocks': 7,
    'encoder-block-type': BlockType.Unet.value,
    'decoder-blocks': 7,
    'decoder-block-type': BlockType.Unet.value,
    'discriminator-blocks': 3,
    'discriminator-block-type': BlockType.Conv.value,
    'channels': 64,
    'adam-lr': 1e-3
}

unet_attn_defaults = {
    'enc-loss-weight': 2,
    'adv-loss-weight': 0.001,
    'encoder-blocks': 7,
    'encoder-block-type': BlockType.Unet.value,
    'decoder-blocks': 8,
    'decoder-block-type': BlockType.Conv.value,
    'discriminator-blocks': 3,
    'discriminator-block-type': BlockType.Conv.value,
    'channels': 64,
    'adam-lr': 1e-3
}


