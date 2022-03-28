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
import glob

class OurCelebA(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset
    """
    def __init__(self, root, train, transform=None, n_samples=None, relimgdir='img_align_celeba', identity_file="identity_CelebA.txt", n_classes=100):

        super(OurCelebA, self).__init__()

        self.root = root
        self.imgdir = os.path.join(root, relimgdir)
        self.train = train  # training set (0) / val set (1) 
        self.transform = transform
        self.identity_file = identity_file
        self.n_classes = n_classes

        try:
            self.identity_data = pd.read_csv(os.path.join(root, self.identity_file), sep=" ", header=None)
            self.identity_data.columns = ["image_id","person_id"]
        except FileNotFoundError:
            raise ValueError("Image partition file {:s} does not exist".format(os.path.join(root, self.identity_file)))

        self.filtered_data = self.identity_data.groupby('person_id').filter(lambda x: len(x) >= 30)
        unique_ids = self.filtered_data.person_id.unique()
        print(unique_ids)
        luid = list(unique_ids)
        print("Inital number of classes:", len(luid))

        print("Selecting first ", self.n_classes)

        class_values = []
        train_val_values = []

        train_pt_counts = {}

        for i in self.filtered_data['person_id']:
            current_class = luid.index(i)
            class_values.append(current_class)
            if i not in train_pt_counts:
                train_pt_counts[i] = 1
            else:
                train_pt_counts[i] += 1
            if current_class>=self.n_classes:
                # Skip these classes
                train_val_values.append(-1)
            else:
                if train_pt_counts[i] >25:
                    train_val_values.append(1)
                else:
                    train_val_values.append(0)

        self.filtered_data['class'] = class_values
        self.filtered_data['train_val'] = train_val_values

        self.sel_rows = self.filtered_data.loc[self.filtered_data['train_val']==self.train]

        self.image_names = self.sel_rows.iloc[:,0].to_numpy()
        self.target_labels = self.sel_rows.iloc[:,2].to_numpy() 

        if n_samples is not None:
            self.image_names = self.image_names[:n_samples]
            self.target_labels = self.target_labels[:n_samples]

        self.n_samples = self.image_names.shape[0]
        print("N Samaples:", self.n_samples)
        assert self.n_samples == self.target_labels.shape[0]

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

        return pilimg, target_label


    def __len__(self):
        return self.n_samples




def filter_open(img_paths):
    filter_paths = []
    for path in img_paths:
        img_name = path.split("/")[-1]
        open_close = int(img_name.split("_")[4])
        if open_close ==1:
            filter_paths.append(path)
    return filter_paths


def process_paths(subject_paths, set_type):
    if set_type == 0:
        set_type_images = subject_paths[:300]
    elif set_type == 1:
        set_type_images = subject_paths[300:350]
    elif set_type == 2:
        set_type_images = subject_paths[350:400]
    else:
        print("Set type invalid")
        exit(0)
    return set_type_images

class MRLeye(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset
    """
    def __init__(self, root, train, transform=None):

        super(MRLeye, self).__init__()

        self.root = root
        self.train = train  # training set (0) / val set (1) / test set (2)
        self.transform = transform
        
        all_subjects = glob.glob(self.root+"s0*")

        n_subjects  = 0
        self.sub_id_mapper = {}

        all_image_paths = []
        for sub_path in all_subjects:
            # print(sub_path)
            per_subject_images = glob.glob(sub_path+ "/*.png")
            # print(len(per_subject_images))
            filtered_subject_images = filter_open(per_subject_images)
            # print(len(filtered_subject_images))
            filtered_subject_images.sort()
            if len(filtered_subject_images) >= 400:
                self.sub_id_mapper[sub_path.split('/')[-1]] = n_subjects
                n_subjects += 1
                all_image_paths.extend(process_paths(filtered_subject_images, self.train))
            else:
                pass

        self.n_subjects = n_subjects
        print("Number of unique subjects: ", self.n_subjects)
        print("Subject id mapper:", self.sub_id_mapper)

        self.all_image_paths = all_image_paths
        self.n_samples = len(self.all_image_paths)

    def __getitem__(self, index):
        fname = self.all_image_paths[index]
        # print(fname)
        pilimg = Image.open(fname)
        # img = np.asarray(pilimg)
        # img = np.transpose(img, axes=(2,0,1))
        
        # img = img - np.min(img)
        # img = np.float32(img/np.max(img))

        if self.transform is not None:
            pilimg = self.transform(pilimg)

        target_label = self.sub_id_mapper[fname.split("/")[-1].split("_")[0]]
        # print(target_label)
        return pilimg, target_label


    def __len__(self):
        return self.n_samples