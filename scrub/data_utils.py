# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import copy

import numpy as np
import torch.utils.data
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip

from dataset import MRLeye


def getDatasets(name='mnist', val_also=True, include_indices=None, exclude_indices=None, data_augment=False):

    if name == 'binary-mnist':
        from torchvision.datasets import MNIST

        train_transformation = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ])


        train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transformation)
        train_dataset = LabelSubsetWrapper(train_dataset, which_labels=(0,1))
        
        val_dataset = MNIST(root='./data', train=False, download=True, transform=train_transformation)
        val_dataset = LabelSubsetWrapper(val_dataset, which_labels=(0,1))

    elif name == 'mnist':
        from torchvision.datasets import MNIST

        train_transformation = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transformation)
        val_dataset = MNIST(root='./data', train=False, download=True, transform=train_transformation)


    elif name == 'cifar10':
        from torchvision.datasets import CIFAR10

        if data_augment:
            print("Doing data augmentation")
            train_transformation = Compose(
                [
                    RandomCrop(32, padding=4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        else:
            train_transformation = Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        val_transformation = Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transformation)
        
        val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transformation)

    elif name == 'mrleye':

        train_transformation = Compose([
            Resize((64,64)),
            ToTensor(),
            # Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = MRLeye(root='/home/sourav/Downloads/mrlEyes_2018_01/', train=0, transform=train_transformation)
        val_dataset = MRLeye(root='/home/sourav/Downloads/mrlEyes_2018_01/', train=1, transform=train_transformation)

    elif name == 'celeba_ID':
        from dataset import OurCelebA

        train_dataset = OurCelebA('./data', n_classes=100)

        val_dataset = OurCelebA('./data', n_classes=100)

    elif name == 'vggface':
        from VGGFaceDataset import VGGFaceDataset

        # for training/validation, just use everything we have
        train_dataset = VGGFaceDataset(attr_data_file='dled_vggface.csv', person='', exclude=True)
        val_dataset = VGGFaceDataset(attr_data_file='dled_vggface.csv', person='', exclude=True)

    else:
        error('unknown dataset')
        

    if include_indices is not None or exclude_indices is not None:
        train_dataset = SubsetDataWrapper(train_dataset, include_indices=include_indices, exclude_indices=exclude_indices)

    return train_dataset, val_dataset




class LabelSubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, which_labels=(0, 1)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which labels to use
        """
        super(LabelSubsetWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics
        self.valid_indices = [idx for idx, (x, y) in enumerate(dataset) if y in which_labels]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.valid_indices[idx]]
        assert y in self.which_labels
        new_y = self.which_labels.index(y)
        return x, torch.tensor(new_y, dtype=torch.long)


BinaryDatasetWrapper = LabelSubsetWrapper  # shortcut


class SubsetDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices=None, include_indices=None):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(SubsetDataWrapper, self).__init__()

        if exclude_indices is None:
            assert include_indices is not None
        if include_indices is None:
            assert exclude_indices is not None

        self.dataset = dataset

        if include_indices is not None:
            self.include_indices = include_indices
        else:
            S = set(exclude_indices)
            self.include_indices = [idx for idx in range(len(dataset)) if idx not in S]

        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]

