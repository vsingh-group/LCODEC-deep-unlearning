from torchvision import datasets, transforms
from ..base import BaseDataLoader
from torch.utils import data
import h5py
import numpy as np
import torch


class SupervisedDataset(data.Dataset):
    """docstring for SupervisedDataset"""
    def __init__(self, h5_path, selected_features=None):
        df = h5py.File(h5_path, "r")
        xh = df.get("X")
        yh = df.get("Y")
        self.length = xh.shape[0]        

        if selected_features is not None:
            self.X = np.empty((xh.shape[0], xh.shape[1], len(selected_features)), order="C")
            selected_features = np.sort(selected_features)
            xh.read_direct(self.X, np.s_[:,:,selected_features], None)
        else:
            self.X = np.empty(xh.shape, order="C")
            xh.read_direct(self.X, None, None)

        self.Y = np.empty(yh.shape, order="C")
        yh.read_direct(self.Y, None, None)
        df.close()

        self.X = torch.from_numpy(np.ascontiguousarray(self.X)).type(torch.float32)
        self.Y = torch.from_numpy(self.Y).type(torch.float32)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class SupervisedDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, 
                 validation_split, test_split, num_workers,
                 selected_features=None, seed=0):
        self.data_dir = data_dir
        self.selected_features = selected_features
        self.seed = seed
        self.dataset = SupervisedDataset(data_dir, selected_features)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers, seed=seed)
