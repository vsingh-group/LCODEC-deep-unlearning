from __future__ import print_function
import os
import os.path
import numpy as np
import pandas as pd
import sys
from scipy import ndimage as nd
import torch
import torch.utils.data as data
from PIL import Image

class OurCelebA(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.

    """
    def __init__(self, root, train, target, spurious, transform=None, n_samples=None, relimgdir='img_align_celeba', attr_data_file="list_attr_celeba.csv", partition_file="list_eval_partition.csv"):

        super(OurCelebA, self).__init__()

        self.root = root
        self.imgdir = os.path.join(root, relimgdir)
        self.train = train  # training set (0) / val set (1) / test set (2) 

        self.transform = transform

        self.attr_data_file = attr_data_file
        self.partition_file = partition_file

        self.target = target
        self.spurious = spurious

        try:
            self.attr_data = pd.read_csv(os.path.join(root, self.attr_data_file))
        except FileNotFoundError:
            raise ValueError("Image attribute file {:s} does not exist".format(os.path.join(root, self.attr_data_file)))

        try:
            self.part_data = pd.read_csv(os.path.join(root, self.partition_file))
        except FileNotFoundError:
            raise ValueError("Image partition file {:s} does not exist".format(os.path.join(root, self.partition_file)))

        self.sel_rows = self.part_data.loc[self.part_data['partition']==self.train]
        self.sel_attrs = self.attr_data.loc[self.part_data['partition']==self.train]
        # self.sel_attrs = self.attr_data.loc[self.attr_data['image_id'].isin(self.sel_rows['image_id'])]

        self.image_names = self.sel_rows.iloc[:,0].to_numpy()

        # +1 for skipping imagename
        self.target_labels = self.sel_attrs.iloc[:,self.target+1].to_numpy() 

        # +1 for skipping imagename
        self.spurious = [i+1 for i in self.spurious]
        self.spurious_labels = self.sel_attrs.iloc[:,self.spurious].to_numpy() 

        # Set labels to 0/1 instead of deafult -1/1
        self.target_labels[self.target_labels<0] = 0
        self.spurious_labels[self.spurious_labels<0] = 0
        

        if n_samples is not None:
            posidxs = self.target_labels==1
            negidxs = self.target_labels==0
            posims = self.image_names[posidxs]
            negims = self.image_names[negidxs]
            postargs = self.target_labels[posidxs]
            negtargs = self.target_labels[negidxs]
            posspurs = self.spurious_labels[posidxs]
            negspurs = self.spurious_labels[negidxs]

            self.image_names = np.concatenate((posims[:n_samples//2], negims[:n_samples//2]))
            self.target_labels = np.concatenate((postargs[:n_samples//2], negtargs[:n_samples//2])) #self.target_labels[:n_samples]
            self.spurious_labels = np.concatenate((posspurs[:n_samples//2], negspurs[:n_samples//2])) #self.spurious_labels[:n_samples]


        self.n_samples = self.image_names.shape[0]

        assert self.n_samples == self.target_labels.shape[0] == self.spurious_labels.shape[0]

    def __getitem__(self, index):

        fname = self.image_names[index]
        pilimg = Image.open(os.path.join(self.imgdir, fname))
        # img = np.asarray(pilimg)
        # img = np.transpose(img, axes=(2,0,1))
        
        # img = img - np.min(img)
        # img = np.float32(img/np.max(img))

        if self.transform is not None:
            pilimg = self.transform(pilimg)

        target_label = self.target_labels[index]
        spurious_label = self.spurious_labels[index]

        return pilimg, target_label, spurious_label


    def __len__(self):
        return self.n_samples
