from enum import Enum

class BlockType(Enum):
    Conv = 'Conv'
    Unet = 'Unet'

hidden_defaults = {
    'enc_loss_weight': 0.7,
    'adv_loss_weight': 0.001,
    'encoder_blocks': 4,
    'encoder_block_type': BlockType.Conv.value,
    'decoder_blocks': 8,
    'decoder_block_type': BlockType.Conv.value,
    'discriminator_blocks': 3,
    'discriminator_block_type': BlockType.Conv.value,
    'encoder_channels': 64,
    'decoder_channels': 64,
    'discriminator_channels': 64,
    'adam_lr': 1e-3
}

unet_conv_defaults = {
    'enc_loss_weight': 4,
    'adv_loss_weight': 0.001,
    'encoder_blocks': 7,
    'encoder_block_type': BlockType.Unet.value,
    'decoder_blocks': 8,
    'decoder_block_type': BlockType.Conv.value,
    'discriminator_blocks': 3,
    'discriminator_block_type': BlockType.Conv.value,
    'encoder_channels': 64,
    'decoder_channels': 64,
    'discriminator_channels': 64,
    'use_dropout': False,
    'adam_lr': 1e-3
}

unet_down_defaults = {
    'enc_loss_weight': 2,
    'adv_loss_weight': 0.001,
    'encoder_blocks': 7,
    'encoder_block_type': BlockType.Unet.value,
    'decoder_blocks': 7,
    'decoder_block_type': BlockType.Unet.value,
    'discriminator_blocks': 3,
    'discriminator_block_type': BlockType.Conv.value,
    'encoder_channels': 64,
    'decoder_channels': 64,
    'discriminator_channels': 64,
    'use_dropout': False,   
    'adam_lr': 1e-3
}

unet_attn_defaults = {
    'enc_loss_weight': 2,
    'adv_loss_weight': 0.001,
    'encoder_blocks': 7,
    'encoder_block_type': BlockType.Unet.value,
    'decoder_blocks': 8,
    'decoder_block_type': BlockType.Conv.value,
    'discriminator_blocks': 3,
    'discriminator_block_type': BlockType.Conv.value,
    'encoder_channels': 64,
    'decoder_channels': 64,
    'discriminator_channels': 64,
    'use_dropout': False,
    'adam_lr': 1e-3
}


shared_defaults_local = {
    'data': '/home/ando/source/hidden/data/medium',
    'batch_size': 32,
    'epochs': 300,
    'name': '',
    'job_name': '$$timestamp--$$main-command--$$noise',
    'jobs_folder': '/home/ando/source/hidden/jobs',
    'tb_folder': '/home/ando/source/hidden/tb-logs',
    'size': 128,
    'message': 30,
    'tensorboard': True,
    'enable_fp16': False,
    'device': 'cuda',
    'noise': ''
}


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged = {**merged, **d}
    return merged


def get_defaults(network_type: str):
    if network_type == 'hidden':
        network_defaults = hidden_defaults
    elif network_type == 'unet-conv':
        network_defaults = unet_conv_defaults
    elif network_type == 'unet-attn':
        network_defaults = unet_attn_defaults
    else:
        raise ValueError(f'network_type == "{network_type}" is not supported')
    # if compute_type == 'local':
    #     shared_defaults = shared_defaults_local
    # elif compute_type.startswith('sagemaker'):
    #     shared_defaults = shared_defaults_sagemaker
    # else:
    #     raise ValueError(f'compute_type == {compute_type} is not supported')

    return merge_dicts(network_defaults, shared_defaults_local)


# def get_sagemaker_defaults(network_type: str):
#     if network_type == 'hidden':
#         network_defaults = hidden_defaults
#     elif network_type == 'unet-conv':
#         network_defaults = unet_conv_defaults
#     elif network_type == 'unet-attn':
#         network_defaults = unet_attn_defaults
#     else:
#         raise ValueError(f'network_type == "{network_type}" is not supported')

#     return merge_dicts(network_defaults, shared_defaults_sagemaker)



def get_supported_architectures():
    return ['hidden', 'unet-conv', 'unet-attn']
