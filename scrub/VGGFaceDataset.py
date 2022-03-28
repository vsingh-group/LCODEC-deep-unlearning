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

def getVGGClassLabelFromName(person='Aamir_Khan', dataroot='./data/vggface/', namefile='names.txt'):
    all_names = pd.read_csv(os.path.join(dataroot, namefile), header=None)

    return all_names[all_names[0]==person].index[0]

class VGGFaceDataset(data.Dataset):
    """
        Dataset assumes a trained model, and dataset constructed
        from data scraped in 2021 from images still available from links
        provided by original authors.

        Gets images for person specified

    """
    def __init__(self, dataroot='./data/vggface/', attr_data_file="vggface_metadata.csv", person='Aamir_Khan', exclude=False):

        super(VGGFaceDataset, self).__init__()

        self.dataroot = dataroot
        self.person = person

        self.attr_data_file = attr_data_file

        try:
            self.attr_data = pd.read_csv(os.path.join(self.dataroot, self.attr_data_file), low_memory=False)
        except FileNotFoundError:
            raise ValueError("Image attribute file {:s} does not exist".format(os.path.join(dataroot, self.attr_data_file)))
        except e:
            print(e)

        print('full shape: ', self.attr_data.shape)
        # TODO: cleanup download script so this is binary value, right now loads as string
        self.attr_data = self.attr_data.loc[self.attr_data['MyCuration']==1]
        print('curated shape: ', self.attr_data.shape)

        if exclude==True:
            sampsperclass = 10
        else:
            sampsperclass = 100

        print('Subsetting persons...')
        tmp = None
        for person in self.attr_data['Person'].unique():
            #print(person)
            if tmp is None:
                tmp = self.attr_data.loc[self.attr_data['Person']==person][:sampsperclass]
            else:
                tmp = tmp.append(self.attr_data.loc[self.attr_data['Person']==person][:sampsperclass], ignore_index=True)

        if exclude:
            tmp.to_csv('vggface_metadata_'+person+'_residual.csv')
        else:
            tmp.to_csv('vggface_metadata_'+person+'_scrub.csv')


        # tmp = self.attr_data.loc[:100]
        print('Done.')

        self.attr_data=tmp
        print('subsetted shape: ', self.attr_data.shape)

        # mask = self.attr_data['FilePath'].apply(lambda x: 'jpg' in x)
        # self.attr_data = self.attr_data.loc[mask]

        print('curated jpg shape: ', self.attr_data.shape)

        # flag for all other people or only this one
        if exclude:
            self.person_data = self.attr_data[~(self.attr_data['Person']==self.person)] 
        else:
            self.person_data = self.attr_data[self.attr_data['Person']==self.person]

        #self.person_data = self.person_data[:64]

        self.n_samples = len(self.person_data)
        print(self.n_samples)


    def __getitem__(self, index):

        imgrow = self.person_data.iloc[index]
        pilimg = Image.open(os.path.join(self.dataroot, imgrow['FilePath']))

        #print(pilimg.size)

        # resizing (already done during download)
        # width, height = pilimg.size
        # scalefactor = 256.0 / np.min([width, height])

        # pilimg = pilimg.resize((int(scalefactor*width), int(scalefactor*height)), resample=Image.BILINEAR)
        # print(pilimg.size)

        # cropping (already done during download)
        # # area = (imgrow['BBL'], imgrow['BBT'], imgrow['BBR'], imgrow['BBB'])
        # w, h = pilimg.size
        # cx = int(w/2)
        # cy = int(h/2)
        # area = (cx-112, cy-112, cx+112, cy+112)
        # pilimg = pilimg.crop(area)

        im = np.array(pilimg)

        # hack for grayscale images
        if len(im.shape) == 2:
            tmp = im
            X = np.zeros((224, 224, 3))
            X[:,:,0] = tmp
            X[:,:,1] = tmp
            X[:,:,2] = tmp

            im = X

        

        # rgb flipping (following vggface convention)

        im = torch.Tensor(im).permute(2, 0, 1).view(3, 224, 224)#.double()
        
        #original training normalization (following vggface convention)
        im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(3,1,1)#.double().view(3, 1, 1)

        return im, int(imgrow['Target'])


    def __len__(self):
        return self.n_samples
