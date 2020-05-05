import sys
import os
import logging
import json
from pathlib import Path
from pprint import pformat
import torch

import wm.util.common as common
from wm.noise.noiser import Noiser
from wm.train.tensorboard_logger import TensorBoardLogger
from wm.train.train_model import train
from wm.model.hidden.hidden_model import Hidden
from wm.model.unet.unet_model import UnetModel


class JobManager:
    def __init__(self, args):
        
        self.resume_mode = args.main_command == 'resume'
        if self.resume_mode:
            self.config = json.load(open(os.path.join(args.folder, 'config.json')))
        else:
            self.config = args.__dict__.copy()
            self.config['timestamp'] = common.get_timestamp()
            self.config['noise'] = '+'.join(sorted(self.config['noise'].split('+')))
            self.config['job_name'] = self._job_name()
            self.config['job_folder'] = os.path.join('.', 'jobs', self.config['job_name'])
            if self.config['tensorboard']:
                noise_folder = self.config['noise'] if self.config['noise'] else 'no-noise'
                self.config['tensorboard_folder'] = os.path.join(self.config['tb_folder'], noise_folder, 
                    f'{self.config["timestamp"]}--{self.config["main_command"].lower()}')

        self.tb_logger = None

    def start_or_resume(self):
        if self.config['tensorboard']:
            self.tb_logger = TensorBoardLogger(self.config['tensorboard_folder'])

        self.model = self._create_model()
        if not self.resume_mode:
            self._create_job_folders()
            self._save_config()

        logging.basicConfig(level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(self.config['job_folder'], f'{self.config["job_name"]}.log')),
            logging.StreamHandler(sys.stdout)
        ])
        logging.info(self.model)

        if self.resume_mode:
            checkpoint, checkpoint_file = common.load_last_checkpoint(os.path.join(self.config['job_folder'], 'checkpoints'))
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f'Loaded checkpoint job checkpoint from file: {checkpoint_file}')
            common.model_from_checkpoint(self.model, checkpoint)
            logging.info(f'Training will resume from epoch={start_epoch}')
        else:
            start_epoch = 1
            logging.info(f'Model:\n{str(self.model)}')
            logging.info(f'Configuration: {pformat(self.config, indent=4)}')

        train(model=self.model, job_name=self.config['job_name'], job_folder=self.config['job_folder'], 
            image_size=self.config['size'], train_folder=os.path.join(self.config['data_dir'], 'train'),
            validation_folder=os.path.join(self.config['data_dir'], 'val'), 
            batch_size=self.config['batch_size'], 
            message_length=self.config['message'],
            number_of_epochs=self.config['epochs'], 
            start_epoch=start_epoch)
        
    def _create_job_folders(self):
        job_folder = self.config['job_folder']
        Path(job_folder).mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.join(job_folder, 'checkpoints'))
        os.makedirs(os.path.join(job_folder, 'images'))

    def _job_name(self):
        job_name = self.config['job_name']
        job_name = job_name.replace('$$timestamp', self.config['timestamp'])
        job_name = job_name.replace('$$main-command', self.config['main_command'].lower())
        if self.config['noise']:
            job_name = job_name.replace('$$noise', self.config['noise'])
        else:
            job_name = job_name.replace('--$$noise', '')
        return job_name

    def _save_config(self):
        with open(os.path.join(self.config['job_folder'], 'config.json'), 'w') as f:
            json.dump(self.config, f)


    def _create_model(self):
        if self.config['main_command'].lower() == 'hidden':
            model = Hidden(config=self.config, tb_logger=self.tb_logger)
        elif self.config['main_command'].lower() in ['unet-conv', 'unet-attn', 'unet-down']:
            model = UnetModel(config=self.config, tb_logger=self.tb_logger)
        return model
        
