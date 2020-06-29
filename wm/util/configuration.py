from enum import Enum

class BlockType(Enum):
    Conv = 'Conv'
    Unet = 'Unet'

LOCAL_BATCH_SIZE = 32
LOCAL_ADAM_LR = 1e-03
AWS_BATCH_SIZE = 96


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
    'discriminator_channels': 64
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
    'use_dropout': False
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
    'use_dropout': False
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
    'use_dropout': False
}


shared_defaults_local = {
    'data': '/home/ando/source/hidden/data/medium',
    'batch_size': LOCAL_BATCH_SIZE,
    'epochs': 300,
    'name': '',
    'job_name_template': '$$timestamp--$$main-command--$$noise--$$suffix',
    'job_name_suffix': '',
    'jobs_folder': '/home/ando/source/hidden/jobs',
    'tb_folder': '/home/ando/source/hidden/tb-logs',
    'size': 128,
    'message': 30,
    'tensorboard': True,
    'enable_fp16': False,
    'device': 'cuda',
    'noise': '',
    'adam_lr': LOCAL_ADAM_LR
}

shared_defaults_aws = {
    's3_bucket': 'watermarking-sagemaker',
    'data_prefix': 'data/medium',
    'instance': 'ml.p3.2xlarge',
    'jobs_prefix': 'jobs',
    'tb_prefix': 'tensorboard',
    'use_spots': False,
    'wait_for_job': False,
    'batch_size': AWS_BATCH_SIZE,
    'epochs': 300,
    'job_name_template': '$$timestamp--$$network_name--$$noise--$$suffix',
    'job_name_suffix': '',
    'size': 128,
    'message': 30,
    'tensorboard': True,
    'enable_fp16': False,
    'device': 'cuda',
    'noise': '',
    'adam_lr': LOCAL_ADAM_LR * (AWS_BATCH_SIZE / LOCAL_BATCH_SIZE)

}

def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged = {**merged, **d}
    return merged


def get_defaults(network_type: str, instance_type: str='local'):
    if network_type == 'hidden':
        network_defaults = hidden_defaults
    elif network_type == 'unet-conv':
        network_defaults = unet_conv_defaults
    elif network_type == 'unet-attn':
        network_defaults = unet_attn_defaults
    else:
        raise ValueError(f'network_type == "{network_type}" is not supported')

    if instance_type == 'local':
        shared_defaults = shared_defaults_local
    elif 'p3.2xlarge' in instance_type:
        shared_defaults = shared_defaults_aws
    else:
        raise ValueError(f'instance type = {instance_type} not recognized') 
    return merge_dicts(network_defaults, shared_defaults)


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
