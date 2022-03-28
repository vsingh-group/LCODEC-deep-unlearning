import numpy as np
import torch
from torchvision.utils import make_grid
from ..base import BaseTrainer
import time
import os

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import scipy.interpolate

import matplotlib.pyplot as plt


class TrainerLasso(BaseTrainer):
    """
    Trainer for feature selection using
    convex combination regularization - alternating optimization

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, make_plots=True):
        super().__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.make_plots = make_plots

        # reconstruction loss
        self.train_loss = []
        self.valid_loss = []

        # keep samples for plotting
        self.train_z = None
        self.valid_z = None

        if "l1_coeff" in self.config['loss']:
            self.l1_coeff = self.config['loss']['l1_coeff']
        else:
            self.l1_coeff = 0.0

        # learning rate scheduler type
        if self.lr_scheduler is not None:
            self.lr_type = self.config['lr_scheduler']['type']
        else:
            self.lr_type = None


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            
            if self.l1_coeff > 0:
                inpw = self.model.input_weights.weight
                loss += self.l1_coeff * torch.abs(inpw).sum()

            loss.backward()
            self.optimizer.step()

            # evaluate metrics; logging
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
        
            total_loss += loss.item()

        # log losses separately
        self.train_loss.append(total_loss/len(self.data_loader))

        # log sum of all losses
        log = {
            'loss': total_loss/len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        # for use with e.g. StepLR
        if self.lr_type == 'StepLR':
            self.lr_scheduler.step()

        # make plots, if applicable
        if self.make_plots:
            self._make_plots(save=epoch%10)

        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)
                
                if self.l1_coeff > 0:
                    inpw = self.model.input_weights.weight
                    loss += self.l1_coeff * torch.abs(inpw).sum()

                # evaluate metrics; logging
                total_val_metrics += self._eval_metrics(output, target)

                total_val_loss += loss.item()

        # log losses separately
        self.valid_loss.append(total_val_loss/len(self.valid_data_loader))

        # for use with e.g. ReduceLROnPlateau
        if self.lr_type == 'ReduceLROnPlateau':
            self.lr_scheduler.step(total_val_loss / len(self.valid_data_loader))

        return {
            'val_loss': total_val_loss/len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

 
    def _eval_metrics(self, output, target):
        return_list = []
        for metric in self.metrics:
            return_list.append(metric(output, target))
        return return_list


    def _make_plots(self, save=False):
        '''
            Run %matplotlib notebook
        '''
        if not hasattr(self, 'fig'):
            self.fig, self.ax1 = plt.subplots(figsize=(4,3))
            
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()

        self.ax1.clear()

        # reconstruction loss
        epochs = np.arange(len(self.train_loss))+1
        self.ax1.semilogy(epochs, self.train_loss, '--', label="Training Loss")
        if len(self.valid_loss) > 0:
            self.ax1.semilogy(epochs, self.valid_loss, '--', label="Validation Loss")

        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.legend()
        self.ax1.grid(True, which="both")

        # draw and sleep
        plt.tight_layout()
        self.fig.canvas.draw()
        if save:
            plt.savefig(os.path.join(self.checkpoint_dir, "train_valid_plot.png"))
