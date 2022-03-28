import argparse
import numpy as np
import pandas as pd
import json
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
from tqdm import tqdm

from models import Net, DisCrim
from dataset import OurCelebA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_classwise_accuracy_sum(target_label, all_labels, target_preds, batch_size):
        tl = target_label
        tp = target_preds.max(1)[1]

        spcnts = []
        nspcnts = []
        sp_accs = []
        not_sp_accs = []

        n_spurious = all_labels.shape[1]
        for i in range(n_spurious):
                g = all_labels[:,i]

                ones = g.nonzero().squeeze()
                zeros = (g==0).nonzero().squeeze()
                
                temp = ones.view(1,-1).shape[1] + zeros.view(1,-1).shape[1]
                assert temp == batch_size

                spcnts.append(ones.view(1,-1).shape[1])
                nspcnts.append(zeros.view(1,-1).shape[1])
 
                sp_acc = (tp[ones]==tl[ones]).float().sum().item()
                sp_accs.append(sp_acc)

                not_sp_acc = (tp[zeros]==tl[zeros]).float().sum().item()
                not_sp_accs.append(not_sp_acc)
        
        return np.asarray(sp_accs), np.asarray(not_sp_accs), np.asarray(spcnts), np.asarray(nspcnts)

def process_tempdict(tmpdict):
        new_split_dic = {}
        new_split_dic['epoch'] = [tmpdict['epoch']]

        new_split_dic['train_full_acc'] = [tmpdict['train_full_acc']]
        new_split_dic['val_full_acc'] = [tmpdict['val_full_acc']]

        many_val_keys = ['train_with_spur_acc', 'train_without_spur_acc', 'val_with_spur_acc', 'val_without_spur_acc']
        for key in many_val_keys:
                list_of_values = tmpdict[key]
                prefix = "_".join(key.split("_", 2)[:2]) + "_"
                for i in range(len(list_of_values)):
                        new_key = prefix + str(i)
                        new_split_dic[new_key] = [list_of_values[i]]
        
        return new_split_dic

def main(args):
        dirname = "Target_"+args.target+"_noise_"+str(args.noise)+"_train_"+str(args.trainset_size)+"_epoch_"+str(args.epochs)
        if not os.path.isdir(dirname):
                os.makedirs(dirname)

        outnameString = dirname + "/" + args.selectionType + '_' + str(args.reg_strength) + '_' + str(args.run)

        model_dirname = "trained_models/Target_"+args.target
        if not os.path.isdir(model_dirname):
                os.makedirs(model_dirname)

        modelnameString = model_dirname + "/" + args.selectionType + '_' + str(args.reg_strength) + '_' + str(args.run) + ".pt"

        # Saving args separately
        argsnameString = dirname + "/args_" + args.selectionType + '_' + str(args.reg_strength) + '_' + str(args.run) +'.txt'
        with open(argsnameString, 'w') as args_file:
                json.dump(args.__dict__, args_file, indent=2)
        args_file.close()

        attr_data = pd.read_csv("./kaggle_data/celeba/list_attr_celeba.csv")

        # removing image column from indexing
        # this value should be 0 to 39
        target_index = attr_data.columns.get_loc(args.target)-1

        # fname = "celeba_foci_npys/"+str(target_index-1)+"_"+args.target+".npy"
        fname = "noise_level_"+str(args.noise)+"/"+str(target_index)+"_"+args.target+".npy"
        foci_numpy = np.load(fname, allow_pickle=True)
        spurious = foci_numpy[1]

        # From now on the spurious_index will contain the actual spurious features which we are regularizing on
        # Also, note that spurious_index will have the attributes numbered from 0 through 39

        # This will be a separate structure to help facilitate the computation of all attribute wise accs, again numbered from 0 through 39
        all_attributes = [i for i in range(attr_data.shape[1]-1)]

        if args.selectionType == 'None':
                spurious_index = []

        elif args.selectionType == 'All':
                # full set of attrs without target
                spurious_index = [i for i in range(attr_data.shape[1]-1)]
                spurious_index.remove(target_index)

        elif args.selectionType == 'FOCI':
                # load selections
                spurious_index = []
                for i in range(len(spurious)):
                        # remove img col, now between 0 and 39
                        sp_index = attr_data.columns.get_loc(spurious[i])-1
                        spurious_index.append(sp_index)
                spurious_index = spurious_index[:1]

        elif args.selectionType == 'Random':
                # get random subset of same size
                spurious_index = np.random.permutation(attr_data.shape[1]-1)[:len(spurious)]

        else:
                return NotImplementedError

        n_dicrims = 0
        discriminators = []
        for i in range(len(spurious_index)):
                d = DisCrim(args.reg_strength).to(device)
                n_dicrims+=1
                discriminators.append(d)

        model = Net().to(device)
        # model.load_state_dict(torch.load(args.MODEL_FILE))
        feature_extractor = model.feature_extractor
        clf = model.classifier


        train_transformation = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = OurCelebA("./kaggle_data/celeba", train=0, target=target_index, spurious=all_attributes, transform=train_transformation, n_samples=args.trainset_size)
        val_dataset = OurCelebA("./kaggle_data/celeba", train=1, target=target_index, spurious=all_attributes, transform=train_transformation, n_samples=args.valset_size)

        print('numer of train target labels = +1:', sum(train_dataset.target_labels==1))
        print('numer of train target labels = -1:', sum(train_dataset.target_labels==0))
        print('percent of train target labels = +1:', sum(train_dataset.target_labels==1)/float(len(train_dataset.target_labels)))

        print('numer of val target labels = +1:', sum(val_dataset.target_labels==1))
        print('numer of val target labels = -1:', sum(val_dataset.target_labels==0))
        print('percent of val target labels = +1:', sum(val_dataset.target_labels==1)/float(len(val_dataset.target_labels)))


        if args.verbose:
                print("Spurious attrs:", spurious_index)
                print("Number of discriminators: ", n_dicrims)
                print("trainset size: ", len(train_dataset))
                print("valset size: ", len(val_dataset))


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                                          num_workers=1, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                                        num_workers=1, pin_memory=True, shuffle=True)

        discriminator_params = []
        for i in range(n_dicrims):
                discriminator_params += list(discriminators[i].parameters())

        optim = torch.optim.Adam(discriminator_params + list(model.parameters()))

        for epoch in range(1, args.epochs+1):

                tmpdict = {}
                tmpdict['epoch'] = epoch
                
                if args.verbose:
                        print("*"*10+"Training"+"*"*10)

                model.train()
                total_label_accuracy = 0
                total_sp_accuracy = np.zeros(len(all_attributes))
                total_not_sp_accuracy = np.zeros(len(all_attributes))
                total_spcnts = np.zeros(len(all_attributes))
                total_nspcnts = np.zeros(len(all_attributes))

                nsamps = 0
                
                for image, target_label, all_labels in tqdm(train_loader):
                        image = image.to(device)
                        target_label = target_label.to(device)

                        features = feature_extractor(image).view(image.shape[0], -1)
                        target_preds = clf(features)

                        target_loss = F.cross_entropy(target_preds, target_label)

                        loss = target_loss

                        if len(spurious_index)!=0:
                                spurious_labels = torch.index_select(all_labels, 1, torch.tensor(spurious_index))
                                for j in range(n_dicrims):
                                        spurious_l = spurious_labels[:,j].to(device)
                                        spurious_p = discriminators[j](features).squeeze()
                                        spurious_l = spurious_l.type(spurious_p.type())
                                        disc_loss = F.binary_cross_entropy_with_logits(spurious_p, spurious_l)
                                        loss = loss + disc_loss

                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        total_label_accuracy += (target_preds.max(1)[1] == target_label).float().sum().item()
                        temp_sp, temp_nsp, spcnts, nspcnts = calc_classwise_accuracy_sum(target_label, all_labels, target_preds, len(target_label))
                        assert len(all_attributes) == len(temp_sp) == len(temp_nsp) == len(spcnts) == len(nspcnts)

                        total_sp_accuracy += temp_sp
                        total_spcnts += spcnts
                        total_not_sp_accuracy += temp_nsp
                        total_nspcnts += nspcnts

                        nsamps += len(target_label)

                overall_accuracy = total_label_accuracy / nsamps 
                spurious_accuracy = total_sp_accuracy / total_spcnts 
                not_spurious_accuracy = total_not_sp_accuracy / total_nspcnts 

                if args.verbose:
                        tqdm.write(f'EPOCH {epoch:03d}: '
                                   f'overall_accuracy={overall_accuracy:.4f}')
                        print("Spurious accuracies: ", spurious_accuracy)
                        print("Not Spurious accuracies: ", not_spurious_accuracy)


                tmpdict['train_full_acc'] = overall_accuracy
                tmpdict['train_with_spur_acc'] = spurious_accuracy
                tmpdict['train_without_spur_acc'] = not_spurious_accuracy

                torch.save(model.state_dict(), modelnameString)        
                if args.verbose:
                        print("*"*10+"Validating"+"*"*10)
                
                model.eval()
                with torch.no_grad():

                        nsamps = 0
                        total_label_accuracy = 0
                        total_sp_accuracy = np.zeros(len(all_attributes))
                        total_not_sp_accuracy = np.zeros(len(all_attributes))
                        total_spcnts = np.zeros(len(all_attributes))
                        total_nspcnts = np.zeros(len(all_attributes))
                                
                        for image, target_label, all_labels in tqdm(val_loader):
                                image = image.to(device)
                                target_label = target_label.to(device)

                                target_preds = model(image)

                                total_label_accuracy += (target_preds.max(1)[1] == target_label).float().sum().item()
                                temp_sp, temp_nsp, spcnts, nspcnts = calc_classwise_accuracy_sum(target_label, all_labels, target_preds, len(target_label))
                                assert len(all_attributes) == len(temp_sp) == len(temp_nsp) == len(spcnts) == len(nspcnts)

                                total_sp_accuracy += temp_sp
                                total_spcnts += spcnts
                                total_not_sp_accuracy += temp_nsp
                                total_nspcnts += nspcnts

                                nsamps += len(target_label)

                overall_accuracy = total_label_accuracy / nsamps 
                spurious_accuracy = total_sp_accuracy / total_spcnts 
                not_spurious_accuracy = total_not_sp_accuracy / total_nspcnts 


                if args.verbose:
                        tqdm.write(f'EPOCH {epoch:03d}: '
                                   f'Validation overall_accuracy={overall_accuracy:.4f}')
                        print("Validation Spurious accuracies: ", spurious_accuracy)
                        print("Validation Not Spurious accuracies: ", not_spurious_accuracy)

                tmpdict['val_full_acc'] = overall_accuracy
                tmpdict['val_with_spur_acc'] = spurious_accuracy
                tmpdict['val_without_spur_acc'] = not_spurious_accuracy

                processed_dict = process_tempdict(tmpdict)
                
                df = pd.DataFrame(processed_dict)
                if epoch == 1:
                        df.to_csv(outnameString+".csv", mode='a', header=True,  index=False)
                else:
                        df.to_csv(outnameString+".csv", mode='a', header=False,  index=False)


if __name__ == '__main__':
        arg_parser = argparse.ArgumentParser(description='CelebA FOCI Runs')
        arg_parser.add_argument('--batch_size', type=int, default=16)
        arg_parser.add_argument('--trainset_size', type=int, default=30000)
        arg_parser.add_argument('--valset_size', type=int, default=4000)
        arg_parser.add_argument('--epochs', type=int, default=10)

        arg_parser.add_argument('--target', type=str, default="Wearing_Lipstick")
        arg_parser.add_argument('--run', type=int, default=0)
        arg_parser.add_argument('--noise', type=float, default=0.01, help="Noise level for generaing the FOCI selections")
        arg_parser.add_argument('--reg_strength', type=float, default=1.0)
        arg_parser.add_argument('--selectionType', type=str, default='All', choices=['None','All', 'FOCI', 'Random'])

        # arg_parser.add_argument('--MODEL_FILE', type=str, default="None", help='A model in trained_models')
        arg_parser.add_argument('--verbose', type=bool, default=False)
        args = arg_parser.parse_args()
        print(args)
        main(args)
