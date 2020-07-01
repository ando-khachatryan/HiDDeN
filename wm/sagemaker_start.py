import argparse
from argparse import ArgumentParser
import os

import pickle
import json
import logging
import sys

from pprint import pprint, pformat
import util.configuration as cfg
from util.sagemaker_config import EXECUTION_ROLE_ARN
from util.common import create_job_name
import train.job_manager as jobman 
import util.parser as parser
from util.common import get_timestamp

import boto3
import sagemaker
from sagemaker.debugger import TensorBoardOutputConfig, DebuggerHookConfig
from sagemaker.pytorch import PyTorch
    

def main():
    
    supported_commands = cfg.get_supported_architectures()
    parent_parser = argparse.ArgumentParser(usage='|'.join(supported_commands))

    subparsers = parent_parser.add_subparsers(dest='main_command', required=True)
    hidden_parser = subparsers.add_parser('hidden', help='HiDDeN architecture')
    parser.newrun_subparser_sagemaker(hidden_parser, 'hidden')

    unet_conv_parser = subparsers.add_parser('unet-conv', help='Unet encoder with HiDDeN decoder')
    parser.newrun_subparser_sagemaker(unet_conv_parser, 'unet-conv')

    unet_attn_parser = subparsers.add_parser('unet-attn', help='Unet attention encoder with decoder')
    parser.newrun_subparser_sagemaker(unet_attn_parser, 'unet-attn')

    resume_parser = subparsers.add_parser('resume', help='Resume a previous run')
    resume_parser.add_argument("folder", type=str,
                                help='Continue from the last checkpoint in this folder.')

    resume_parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                                help='Turn off TensorBoard logging.')
    resume_parser.set_defaults(tensorboard=True)

    args = parent_parser.parse_args()
    args.noise = '+'.join(sorted(args.noise.split('+')))
    pprint(args.__dict__)

    sagemaker_session = sagemaker.Session()
    job_ts = get_timestamp().replace('.', '-')
    job_name = create_job_name(timestamp=job_ts, network_name=args.main_command, template=args.job_name_template, 
                                suffix=args.job_name_suffix, noise=args.noise)

    params = {}
    for key in ['main-command', 'batch-size', 'epochs', 'size', 'message', 'device', 'enc-loss-weight',
                'adv-loss-weight', 'encoder-blocks', 'encoder-channels', 'encoder-block-type', 
                'decoder-blocks', 'decoder-channels', 'decoder-block-type', 'discriminator-blocks',
                'discriminator-channels', 'discriminator-block-type', 'adam-lr', 'noise']:
        params[key] = args.__dict__[key.replace('-', '_')]

    params['job-name'] = job_name
    params['tensorboard'] = 1 * args.tensorboard
    params['enable-fp16'] = 1 * args.enable_fp16
    print(f'-'*80)
    pprint(params)
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f's3://{args.s3_bucket}/tensorboard/{args.noise}',
        container_local_output_path=f'/tb-logs'
    )

    # print(type(tensorboard_output_config))
    # print(isinstance(tensorboard_output_config, DebuggerHookConfig))
    # pprint(tensorboard_output_config.__dict__)


    train_data = sagemaker.session.s3_input(
    s3_data=f's3://{args.s3_bucket}/{args.data_prefix}/train',
    distribution='FullyReplicated',
    s3_data_type='S3Prefix')

    test_data = sagemaker.session.s3_input(
        s3_data=f's3://{args.s3_bucket}/{args.data_prefix}/val',
        distribution='FullyReplicated',
        s3_data_type='S3Prefix')
    
    job = PyTorch(entry_point='sm-entry.py', 
              source_dir='.', 
              framework_version='1.4.0',
              train_instance_count=1,
              train_instance_type=args.instance, 
              hyperparameters=params,
              role=EXECUTION_ROLE_ARN,
              output_path=f's3://{args.s3_bucket}/jobs/{job_name}/output-path',
              base_job_name=args.main_command,
              code_location=f's3://{args.s3_bucket}/jobs/{job_name}'
              # ,checkpoint_s3_uri=f's3://{S3_BUCKET}/jobs/{job_name}/checkpoints'
              ,tensorboard_output_config=tensorboard_output_config
              ,train_use_spot_instances=args.use_spots
              # ,train_max_wait=ONE_HOUR
              )
    job.fit(inputs={'train': train_data, 'test': test_data}, wait=args.wait_for_job)

if __name__ == "__main__":
    main()