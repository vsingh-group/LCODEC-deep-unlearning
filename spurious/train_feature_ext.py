import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from models import Net
from dataset import OurCelebA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true, _ in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def main(args):
    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = OurCelebA("./kaggle_data/celeba", train=0, target=36, spurious=[3,4,6], transform=train_transformation)
    print(len(train_dataset))
    val_dataset = OurCelebA("./kaggle_data/celeba", train=1, target=36, spurious=[3,4,6], transform=train_transformation)
    print(len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=1, pin_memory=True)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=16)
    arg_parser.add_argument('--epochs', type=int, default=30)
    args = arg_parser.parse_args()
    main(args)