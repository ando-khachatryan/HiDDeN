import boto3
import sagemaker
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch
import os
from pathlib import Path
from util.sagemaker_config import EXECUTION_ROLE_ARN, S3_BUCKET
import util.configuration as cfg
from util.common import get_timestamp
from pprint import pprint
import logging

sagemaker_session = sagemaker.Session()
data_prefix = 'data/small'
# data_prefix = 'data/medium'

# instance_type = 'local'
instance_type = 'ml.m5.xlarge'
# instance_type = 'ml.p3.2xlarge'

job_ts = get_timestamp().replace('.', '-')
job_name = f'hidden--{job_ts}'
tb_logs_dir = f'/tb-logs'

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001
BATCH_SIZE = 128

args = {
 'main-command': 'hidden',
 'adam-lr': DEFAULT_LR * (BATCH_SIZE/DEFAULT_BATCH_SIZE),
 'adv-loss-weight': 0.001,
 'batch-size': BATCH_SIZE,
 'decoder-block-type': 'Conv',
 'decoder-blocks': 8,
 'decoder-channels': 64,
 'device': 'cuda',
 'discriminator-block-type': 'Conv',
 'discriminator-blocks': 3,
 'discriminator-channels': 64,
 'enc-loss-weight': 0.7,
 'encoder-block-type': 'Conv',
 'encoder-blocks': 4,
 'encoder-channels': 64,
 'epochs': 50,
 'message': 30,
 'size': 128,
 'tensorboard': 1,
 'tensorboard-folder': tb_logs_dir}

if instance_type in ['ml.m5.xlarge', 'local']:
     args['device'] = 'cpu'

args['noise'] = 'jpeg()+blur(2.0)'
args['job-name'] = job_name

train_data = sagemaker.session.s3_input(
    s3_data=f's3://{S3_BUCKET}/{data_prefix}/train',
    distribution='FullyReplicated',
    s3_data_type='S3Prefix')

test_data = sagemaker.session.s3_input(
    s3_data=f's3://{S3_BUCKET}/{data_prefix}/val',
    distribution='FullyReplicated',
    s3_data_type='S3Prefix')

print(f'job base folder is: {job_name}')
print(f'instance type: {instance_type}')


tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=f's3://{S3_BUCKET}/tensorboard',
    container_local_output_path=tb_logs_dir
)
ONE_HOUR = 60 * 60

job = PyTorch(entry_point='sm-entry.py', 
              source_dir='.', 
              framework_version='1.4.0',
              train_instance_count=1,
              train_instance_type=instance_type, 
              hyperparameters=args,
              role=EXECUTION_ROLE_ARN,
              output_path=f's3://{S3_BUCKET}/jobs/{job_name}/output-path',
            #   base_job_name=job_name, 
              code_location=f's3://{S3_BUCKET}/jobs/{job_name}'
              # ,checkpoint_s3_uri=f's3://{S3_BUCKET}/jobs/{job_name}/checkpoints'
              ,tensorboard_output_config=tensorboard_output_config
            #   ,container_log_level=logging.INFO
              # ,train_use_spot_instances=True
              # ,train_max_wait=ONE_HOUR
              )

job.fit(inputs={'train': train_data, 'test': test_data}, wait=True)
# job.fit(inputs={'train': train_data})
