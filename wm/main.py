import argparse
from argparse import ArgumentParser
import os

import pickle
import json
import logging
import sys

import torch
from pprint import pprint, pformat
import util.configuration as cfg
import train.job_manager as jobman 
import util.parser as parser
    

def main():
    
    supported_commands = cfg.get_supported_architectures()
    supported_commands.append('resume')
    parent_parser = argparse.ArgumentParser(usage='|'.join(supported_commands))

    subparsers = parent_parser.add_subparsers(dest='main_command', required=True)
    hidden_parser = subparsers.add_parser('hidden', help='HiDDeN architecture')
    parser.create_new_run_subparser(hidden_parser, 'hidden')

    unet_conv_parser = subparsers.add_parser('unet-conv', help='Unet encoder with HiDDeN decoder')
    parser.create_new_run_subparser(unet_conv_parser, 'unet-conv')

    unet_attn_parser = subparsers.add_parser('unet-attn', help='Unet attention encoder with decoder')
    parser.create_new_run_subparser(unet_attn_parser, 'unet-attn')

    resume_parser = subparsers.add_parser('resume', help='Resume a previous run')
    resume_parser.add_argument("folder", type=str,
                                help='Continue from the last checkpoint in this folder.')

    resume_parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                                help='Turn off TensorBoard logging.')
    resume_parser.set_defaults(tensorboard=True)

    args = parent_parser.parse_args()
    job = jobman.JobManager(args)
    print('-'*80)
    job.start_or_resume()
        

if __name__ == "__main__":
    main()