import glob
import os
import json
import argparse
import torch

import numpy as np
import matplotlib.pyplot as plt
import pickle

from .utils import Logger
from . import model as module_model
from . import data_loader as module_data
from .evaluation import loss as module_loss
from .evaluation import metric as module_metric
from . import trainer as module_trainer


class ModelManager:
    """ ModelManager keeps track of model training, validation, and test """
    def __init__(self, config_path, make_plots=True, is_resume=False):
        self.config, self.resume = self.initialize(config_path, is_resume)
        self.train_logger = Logger()

        # build model
        self.model = self.get_instance(module_model, 'arch', self.config)

        # set up data loader instances
        self.data_loader = self.get_instance(module_data, 'data_loader', self.config)
        self.valid_data_loader = self.data_loader.split_validation()
        self.test_data_loader = self.data_loader.split_test()

        # set up loss functions and metrics
        self.loss = getattr(module_loss, self.config['loss']['type'])
        self.metrics = [getattr(module_metric, met) for met in self.config['metrics']]

        # build optimizer and learning rate scheduler
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.get_instance(torch.optim, 'optimizer', self.config, self.trainable_params)
        self.lr_scheduler = self.get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config, self.optimizer)

        # initialize trainer
        self.trainer = getattr(module_trainer,self.config['trainer']['type'])(
            self.model, self.loss, self.metrics, self.optimizer, 
            resume=self.resume,
            config=self.config,
            data_loader=self.data_loader,
            valid_data_loader=self.valid_data_loader,
            lr_scheduler=self.lr_scheduler,
            train_logger=self.train_logger,
            make_plots=make_plots)


    def get_instance(self, module, name, config, *args):
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])


    def initialize(self, path, is_resume):
        if is_resume:
            config = torch.load(path)['config']
            return config, path
        else:
            config = json.load(open(path))
            return config, False


    def train(self):
        self.model.train()
        self.trainer.train()


    def clean_checkpoint_files(self):
        for f in glob.glob(os.path.join(self.trainer.checkpoint_dir, "checkpoint-*.pth")):
            os.remove(f)


    def load_model(self, checkpoint_file="model_best.pth", fullpath=False):
        if fullpath:
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(os.path.join(self.trainer.checkpoint_dir, checkpoint_file))
        
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        return device


    def process_numpy(self, input_data, checkpoint_file="model_best.pth", fullpath=False):
        device = self.load_model(checkpoint_file, fullpath=fullpath)

        input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

        outputs = self.model(input_data)
        if type(outputs)==tuple:
            return [output.detach().cpu().numpy() for output in outputs]
        else:
            return outputs.detach().cpu().numpy()
