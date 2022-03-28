"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse
import numpy as np
import pandas as pd

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

def calc_classwise_accuracy(target_label, spurious_labels, target_preds):
    tl = target_label
    tp = target_preds.max(1)[1]

    sp_accs = []
    not_sp_accs = []

    n_spurious = spurious_labels.shape[1]
    for i in range(n_spurious):
        g = spurious_labels[:,i]
        ones = g.nonzero().squeeze()
        zeros = (g==0).nonzero().squeeze()
        sp_acc = (tp[ones]==tl[ones]).float().mean().item()
        not_sp_acc = (tp[zeros]==tl[zeros]).float().mean().item()
        sp_accs.append(sp_acc)
        not_sp_accs.append(not_sp_acc)

    return np.asarray(sp_accs), np.asarray(not_sp_accs)


def main(args):
    attr_data = pd.read_csv("./kaggle_data/celeba/list_attr_celeba.csv")
    target_index = attr_data.columns.get_loc(args.target)
    fname = "celeba_foci_npys/"+str(target_index-1)+"_"+args.target+".npy"
    print(fname)
    foci_numpy = np.load(fname)
    print("FOCI")
    print(foci_numpy)
    spurious = foci_numpy[1]
    spurious_index = []
    for i in range(len(spurious)):
        sp_index = attr_data.columns.get_loc(spurious[i])
        spurious_index.append(sp_index)

    spurious_index = spurious_index[:1]
    print("Spurious attrs:", spurious_index)

    model = Net().to(device)
    # model.load_state_dict(torch.load(args.MODEL_FILE))

    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = OurCelebA("./kaggle_data/celeba", train=0, target=target_index, spurious=spurious_index, transform=train_transformation, n_samples=150)
    print(len(train_dataset))
    val_dataset = OurCelebA("./kaggle_data/celeba", train=1, target=target_index, spurious=spurious_index, transform=train_transformation, n_samples=None)
    print(len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=1, pin_memory=True)

    optim = torch.optim.Adam(model.parameters())

    for epoch in range(1, args.epochs+1):
        
        print("*"*10+"Training"+"*"*10)

        model.train()
        total_label_accuracy = 0
        total_sp_accuracy = 0
        total_not_sp_accuracy = 0
        
        for image, target_label, spurious_labels in tqdm(train_loader):
            image = image.to(device)
            target_label = target_label.to(device)

            target_preds = model(image)

            target_loss = F.cross_entropy(target_preds, target_label)

            loss = target_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_label_accuracy += (target_preds.max(1)[1] == target_label).float().mean().item()
            temp_sp, temp_nsp = calc_classwise_accuracy(target_label, spurious_labels, target_preds)
            total_sp_accuracy += temp_sp
            total_not_sp_accuracy += temp_nsp

        overall_accuracy = total_label_accuracy / len(train_loader)
        spurious_accuracy = total_sp_accuracy / len(train_loader)
        not_spurious_accuracy = total_not_sp_accuracy / len(train_loader)

        tqdm.write(f'EPOCH {epoch:03d}: '
                   f'overall_accuracy={overall_accuracy:.4f}')
        print("Spurious accuracies: ", spurious_accuracy)
        print("Not Spurious accuracies: ", not_spurious_accuracy)

        torch.save(model.state_dict(), 'trained_models/plain.pt')

        
        print("*"*10+"Validating"+"*"*10)
        
        model.eval()
        with torch.no_grad():

            total_label_accuracy = 0
            total_sp_accuracy = 0
            total_not_sp_accuracy = 0
            
            for image, target_label, spurious_labels in tqdm(val_loader):
                image = image.to(device)
                target_label = target_label.to(device)

                target_preds = model(image)

                total_label_accuracy += (target_preds.max(1)[1] == target_label).float().mean().item()
                temp_sp, temp_nsp = calc_classwise_accuracy(target_label, spurious_labels, target_preds)
                total_sp_accuracy += temp_sp
                total_not_sp_accuracy += temp_nsp

            overall_accuracy = total_label_accuracy / len(val_loader)
            spurious_accuracy = total_sp_accuracy / len(val_loader)
            not_spurious_accuracy = total_not_sp_accuracy / len(val_loader)

            tqdm.write(f'EPOCH {epoch:03d}: '
                       f'overall_accuracy={overall_accuracy:.4f}')
            print("Spurious accuracies: ", spurious_accuracy)
            print("Not Spurious accuracies: ", not_spurious_accuracy)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('--target', type=str, default="Wearing_Lipstick")
    arg_parser.add_argument('--MODEL_FILE', type=str, default="trained_models/source.pt", help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=16)
    arg_parser.add_argument('--epochs', type=int, default=15)
    args = arg_parser.parse_args()
    main(args)
