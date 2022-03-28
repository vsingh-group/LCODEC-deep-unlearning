import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

from ..base import BaseDataLoader


class NpSupervisedDataset(data.Dataset):
    """docstring for SupervisedDataset"""
    def __init__(self, x_path, y_path, selected_features=None, bit16=False):
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        
        assert x_data.shape[0] == y_data.shape[0]
        self.length = x_data.shape[0]
        self.dtype = torch.float16 if bit16 else torch.float32

        if selected_features is not None:
            selected_features = np.sort(selected_features)
            self.X = torch.from_numpy(np.ascontiguousarray(x_data[:,:,selected_features])).type(self.dtype)
        else:
            self.X = torch.from_numpy(np.ascontiguousarray(x_data)).type(self.dtype)

        self.Y = torch.from_numpy(y_data).type(self.dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class NpSupervisedDataLoader(BaseDataLoader):
    def __init__(self, x_path, y_path, batch_size, shuffle, 
                 validation_split, test_split, num_workers,
                 selected_features=None, bit16=False, seed=0):       
        self.selected_features = selected_features
        self.seed = seed
        self.dataset = NpSupervisedDataset(x_path, y_path, selected_features, bit16)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers, seed=seed)
