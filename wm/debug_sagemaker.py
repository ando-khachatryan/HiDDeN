import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import os
from pathlib import Path

sagemaker_session = sagemaker.Session()
bucket = 'watermarking-sagemaker'
role = 'arn:aws:iam::192442179081:role/service-role/AmazonSageMaker-ExecutionRole-20180214T113521'

training_job_name = 'pytorch-training-2020-06-17-14-45-31-406'
client = boto3.client('sagemaker')
response = client.describe_training_job(
    TrainingJobName=training_job_name
)

print(response)
