from torchvision import datasets, transforms
from ..base import BaseDataLoader
from torch.utils import data
import h5py
import numpy as np
import torch


class TimeseriesDataset(data.Dataset):
    """docstring for BackBlazeDataset"""
    def __init__(self, h5_path, feat_list=None, time_trimmed=0, tia=0):
        df = h5py.File(h5_path, "r")
        xh = df.get("X")
        yh = df.get("Y")
        assert tia >= 0
        self.length = xh.shape[0]

        if feat_list is not None:
            self.X = np.empty(
                (xh.shape[0], xh.shape[1]-tia-time_trimmed, len(feat_list)), 
                order="C")
            if tia > 0:
                xh.read_direct(self.X, np.s_[:,time_trimmed:-tia,feat_list], None)
            else:
                xh.read_direct(self.X, np.s_[:,time_trimmed:,feat_list], None)
        else:
            self.X = np.empty(
                (xh.shape[0], xh.shape[1]-tia, xh.shape[2]), 
                order="C")
            if tia > 0:
                xh.read_direct(self.X, np.s_[:,:-tia,:], None)
            else:
                xh.read_direct(self.X, np.s_[:,:,:], None)

        self.Y = np.empty(yh.shape, order="C")
        yh.read_direct(self.Y, None, None)
        df.close()

        self.X = torch.from_numpy(
            np.ascontiguousarray(self.X)).type(torch.float32)
        self.Y = torch.from_numpy(self.Y).type(torch.float32)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index,...], self.Y[index]


class TimeseriesDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, 
                 validation_split, test_split, num_workers, 
                 feat_list=None, time_trimmed=0, tia=0, training=True, seed=0):
        self.data_dir = data_dir
        self.feat_list = feat_list
        self.time_trimmed = time_trimmed
        self.tia = tia
        self.seed = seed
        self.dataset = TimeseriesDataset(data_dir, feat_list, time_trimmed, tia)
        super().__init__(
            self.dataset, batch_size, shuffle, 
            validation_split,test_split, num_workers, seed=seed)
