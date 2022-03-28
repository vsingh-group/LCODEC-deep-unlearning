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


class TrainerPushPull(BaseTrainer):
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

        if "plotz" in self.config['trainer']:
            self.plotz = self.config['trainer']['plotz']
        else:
            self.plotz = False

        if "reg_coeff" in self.config['loss']:
            self.reg_coeff = self.config['loss']['reg_coeff']
        else:
            self.reg_coeff = 0.0

        if "l1_coeff" in self.config['loss']:
            self.l1_coeff = self.config['loss']['l1_coeff']
        else:
            self.l1_coeff = 0.0

        # learning rate scheduler type
        if self.lr_scheduler is not None:
            self.lr_type = self.config['lr_scheduler']['type']
        else:
            self.lr_type = None


    def _onestep_loss(self, data, target, enable_block_drop=True):
        if self.model.training:
            self.optimizer.zero_grad()

        # get z with no dropout, apply dropout for likelihood term only
        z = self.model._encode(data, apply_block_drop=enable_block_drop)
        output = self.model._predict(z)

        # if enable_block_drop:
        #     output = self.model._predict(z)
        # else:
        #     output = self.model._predict(z/(1-self.model.p_drop))

        loss = self.loss(output, target)

        # l1 reg
        if self.l1_coeff > 0:
            loss += self.l1_coeff*torch.mean(torch.norm(z, p=1, dim=1))

        # reshape reg
        if self.reg_coeff > 0:
            num_features = self.model.num_features
            batch_size = z.shape[0]
            
            # loop through features that have not been dropped out
            for i in range(num_features):
                # skip if feature has been dropped out
                if not torch.all(z[:,:,i]==0).item(): 
                    # shuffle i-th feature
                    z_tilde = Variable(z.new(z.shape))
                    z_tilde[:] = z.clone()
                    z_tilde[:,:,i] = z[np.random.permutation(batch_size),:,i].clone()

                    # compute squared euclidean distance
                    distance = ((z[:,:,i] - z_tilde[:,:,i])**2).sum(dim=1)
                    # distance = (z[:,:,i] - z_tilde[:,:,i]).norm(p=2, dim=1)

                    # compute jeffreys divergence
                    jeffreys = self.model.divergence(z, z_tilde).view(-1)

                    # compute loss
                    loss += self.reg_coeff*torch.sum((distance-jeffreys).abs())/batch_size

        if self.model.training:
            loss.backward()
            self.optimizer.step()

        return loss, output, z


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

            # loss,output,z = self._recon_loss(data, target, n_runs=self.predict_runs)
            loss, output, z = self._onestep_loss(data, target, enable_block_drop=True)

            # evaluate metrics; logging
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

            # sample z values for plotting
            if batch_idx == 0 and self.plotz:
                self.train_z = z.detach().cpu().numpy()
        
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

                # loss, output, z = self._recon_loss(data, target)
                loss, output, z = self._onestep_loss(data, target, enable_block_drop=True)

                # evaluate metrics; logging
                total_val_metrics += self._eval_metrics(output, target)

                if batch_idx == 0 and self.plotz:
                    self.valid_z = z.detach().cpu().numpy()

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
            if self.plotz:
                self.fig,(self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8,3))
            else:
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

        if self.plotz:
            self.ax2.clear()
            idx = np.random.choice(self.train_z.shape[2])
            if self.train_z.shape[1] > 1:
                self.ax2.scatter(self.train_z[:,0], self.train_z[:,1], label="Training Z")
            else:
                self.ax2.hist(self.train_z[:,0,idx], density=True, label="Training Z: dim: %d"%idx)

            if len(self.valid_loss) > 0:
                if self.valid_z.shape[1] > 1:
                    self.ax2.scatter(self.valid_z[:,0], self.valid_z[:,1], label="Validation Z")
                else:
                    self.ax2.hist(self.valid_z[:,0,idx], density=True, label="Validation Z: dim: %d"%idx)

            self.ax2.legend()

        # draw and sleep
        plt.tight_layout()
        self.fig.canvas.draw()
        if save:
            plt.savefig(os.path.join(self.checkpoint_dir, "train_valid_plot.png"))
