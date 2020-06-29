import os
from argparse import ArgumentParser
import util.configuration as cfg


def create_new_run_subparser(newrun_parser: ArgumentParser, network_type: str):
    recognized_networks = ['hidden', 'unet-conv', 'unet-attn']
    if network_type not in recognized_networks:
        raise ValueError(f'network type must be one of {recognized_networks}, instead it is "{network_type}".') 
        
    defaults = cfg.get_defaults(network_type=network_type, instance_type='local')

    newrun_parser.add_argument('--data',  type=str, required=False, help='The input data directory')
    # newrun_parser.add_argument('--val-dir', default=defaults['val-dir'], type=str, help='Validation images folder')
    newrun_parser.add_argument('--batch-size', required=False, type=int, help='The batch size.')
    newrun_parser.add_argument('--epochs',  required=False, type=int, help='Number of epochs to run the simulation.')

    newrun_parser.add_argument('--name', required=False, type=str, help='The name of the experiment.')
    newrun_parser.add_argument('--job-name_template', required=False, type=str, help='String representation for the experiment name')
    newrun_parser.add_argument('--job_name_suffix', required=False, type=str, help='Suffix string to add to the job name')
    newrun_parser.add_argument('--jobs-folder', required=False, type=str, help='The root folder of experiments')
    newrun_parser.add_argument('--tb-folder', required=False, type=str, help='The root tensorboard folder')

    newrun_parser.add_argument('--size', default=defaults['size'], type=int,
                                help='The size of the images (images are square so this is height and width).')
    newrun_parser.add_argument('--message', default=defaults['message'], type=int, help='The length in bits of the watermark.')
    newrun_parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                                help='Turn off TensorBoard logging.')
    newrun_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    newrun_parser.add_argument('--device', required=False, type=str, choices=['cuda', 'cpu'], help='The device, cuda|cpu.')

    newrun_parser.add_argument('--enc-loss-weight', required=False, type=float,
                                help='The weight of encoder loss in the overall loss function')
    newrun_parser.add_argument('--adv-loss-weight', required=False,  type=float,
                                help='The weight of the adversarial loss in the overall loss function')

    newrun_parser.add_argument('--encoder-blocks', required=False,  type=str, help='Number of blocks in the encoder.')
    newrun_parser.add_argument('--encoder-channels', required=False,  type=int, help='Number of inner channels in encoder blocks.')
    newrun_parser.add_argument('--encoder-block-type', required=False, choices=['Conv', 'Unet'], type=str, help='Encoder block type.')

    newrun_parser.add_argument('--decoder-blocks', required=False, type=int, help='Number of blocks in the decoder')
    newrun_parser.add_argument('--decoder-channels', required=False, type=int, help='Number of channels in decoder blocks.')
    newrun_parser.add_argument('--decoder-block-type', required=False, choices=['Conv', 'Unet'], 
                                type=str, help='Decoder block type.')

    newrun_parser.add_argument('--discriminator-blocks', required=False, type=str, help='Number of blocks in the discriminator.')
    newrun_parser.add_argument('--discriminator-channels', required=False, type=str, help='Number of channels in discriminator blocks.')
    newrun_parser.add_argument('--discriminator-block-type', required=False, choices=['Conv', 'Unet'], type=str, help='discriminator block type')

    newrun_parser.add_argument('--adam-lr', required=False, type=float, help='Learning rate of Adam')

    newrun_parser.add_argument('--noise', type=str, required=False,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout(0.55, 0.6)'")

    # newrun_parser.add_argument('--sagemaker', type=str, default='', help='Sagemaker instance type. if not empty, will run the code on sagemaker.')
    if network_type.startswith('unet'):
        newrun_parser.add_argument('--use-dropout', dest='use_dropout', action='store_true', help='enable dropout in unet encoder')
    
    newrun_parser.set_defaults(**defaults)


def newrun_subparser_sagemaker(newrun_parser: ArgumentParser, network_type: str, instance_type: str='ml.p3.2xlarge'):
    recognized_networks = ['hidden', 'unet-conv', 'unet-attn']
    if network_type not in recognized_networks:
        raise ValueError(f'network type must be one of {recognized_networks}, instead it is "{network_type}".') 
    
    supported_instance_types = ['local', 'ml.m5.xlarge', 'ml.p3.2xlarge']
    if instance_type not in supported_instance_types:
        raise ValueError(f'instance type must be on of {supported_instance_types}, instead it is {instance_type}')
    
    defaults = cfg.get_defaults(network_type=network_type, instance_type=instance_type)
    # newrun_parser.add_argument('--data',  type=str, required=False, help='The input data directory')
    # newrun_parser.add_argument('--val-dir', default=defaults['val-dir'], type=str, help='Validation images folder')
    newrun_parser.add_argument('--s3-bucket', type=str, required=False, help='S3 bucket.')
    newrun_parser.add_argument('--data-prefix', type=str, required=False, help='Data prefix on S3.')
    newrun_parser.add_argument('--instance', type=str, required=False, help='The AWS instance type to use')
    newrun_parser.add_argument('--use-spot-instances', action='store_true', dest='use_spots', help='Use spot instances for training')
    newrun_parser.add_argument('--wait', action='store_true', dest='wait_for_job', help='Wait for job to complete on sagemaker instead of exiting upon job start')

    newrun_parser.add_argument('--batch-size', required=False, type=int, help='The batch size.')
    newrun_parser.add_argument('--epochs',  required=False, type=int, help='Number of epochs to run the simulation.')

    newrun_parser.add_argument('--job_name_suffix', required=False, type=str, help='The name of the experiment.')
    newrun_parser.add_argument('--job-name_template', required=False, type=str, help='String representation for the experiment name')
    newrun_parser.add_argument('--jobs-prefix', required=False, type=str, help='S3 key of the root folder of experiments')
    newrun_parser.add_argument('--tb-prefix', required=False, type=str, help='S3 key of the root for tensorboard data')

    newrun_parser.add_argument('--size', default=defaults['size'], type=int,
                                help='The size of the images (images are square so this is height and width).')
    newrun_parser.add_argument('--message', default=defaults['message'], type=int, help='The length in bits of the watermark.')
    newrun_parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                                help='Turn off TensorBoard logging.')
    newrun_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    newrun_parser.add_argument('--device', required=False, type=str, choices=['cuda', 'cpu'], help='The device, cuda|cpu.')

    newrun_parser.add_argument('--enc-loss-weight', required=False, type=float,
                                help='The weight of encoder loss in the overall loss function')
    newrun_parser.add_argument('--adv-loss-weight', required=False,  type=float,
                                help='The weight of the adversarial loss in the overall loss function')

    newrun_parser.add_argument('--encoder-blocks', required=False,  type=str, help='Number of blocks in the encoder.')
    newrun_parser.add_argument('--encoder-channels', required=False,  type=int, help='Number of inner channels in encoder blocks.')
    newrun_parser.add_argument('--encoder-block-type', required=False, choices=['Conv', 'Unet'], type=str, help='Encoder block type.')

    newrun_parser.add_argument('--decoder-blocks', required=False, type=int, help='Number of blocks in the decoder')
    newrun_parser.add_argument('--decoder-channels', required=False, type=int, help='Number of channels in decoder blocks.')
    newrun_parser.add_argument('--decoder-block-type', required=False, choices=['Conv', 'Unet'], 
                                type=str, help='Decoder block type.')

    newrun_parser.add_argument('--discriminator-blocks', required=False, type=str, help='Number of blocks in the discriminator.')
    newrun_parser.add_argument('--discriminator-channels', required=False, type=str, help='Number of channels in discriminator blocks.')
    newrun_parser.add_argument('--discriminator-block-type', required=False, choices=['Conv', 'Unet'], type=str, help='discriminator block type')

    newrun_parser.add_argument('--adam-lr', required=False, type=float, help='Learning rate of Adam')

    newrun_parser.add_argument('--noise', type=str, required=False,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout(0.55, 0.6)'")

    # newrun_parser.add_argument('--sagemaker', type=str, default='', help='Sagemaker instance type. if not empty, will run the code on sagemaker.')
    if network_type.startswith('unet'):
        newrun_parser.add_argument('--use-dropout', dest='use_dropout', action='store_true', help='enable dropout in unet encoder')

    newrun_parser.set_defaults(**defaults)
    

def create_local_run_parser():
    parent_parser = ArgumentParser(usage='hidden|unet-conv|unet-down|resume')

    subparsers = parent_parser.add_subparsers(dest='main_command', required=True)
    hidden_parser = subparsers.add_parser('hidden', help='HiDDeN architecture')
    create_new_run_subparser(hidden_parser, 'hidden')

    unet_conv_parser = subparsers.add_parser('unet-conv', help='Unet encoder with HiDDeN decoder')
    create_new_run_subparser(unet_conv_parser, 'unet-conv')

    # unet_down_parser = subparsers.add_parser('unet-down', help='Unet encoder with downsampling decoder')
    # create_new_run_subparser(unet_down_parser, 'unet-down')
    unet_attn_parser = subparsers.add_parser('unet-attn', help='Unet attention encoder with decoder')
    create_new_run_subparser(unet_attn_parser, 'unet-attn')

    resume_parser = subparsers.add_parser('resume', help='Resume a previous run')
    resume_parser.add_argument("folder", type=str,
                                help='Continue from the last checkpoint in this folder.')

    resume_parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                                help='Turn off TensorBoard logging.')
    resume_parser.set_defaults(tensorboard=True)



