import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Subset

from data_utils import getDatasets
from nn_utils import do_epoch
from grad_utils import getGradObjs

from nn_utils import manual_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    manual_seed(args.train_seed)
    outString = 'trained_models/'+args.dataset+"_"+args.model+'_ntp_'+str(args.used_training_size)+'_seed_' + str(args.train_seed)+'_epochs_' + str(args.epochs)+'_lr_' + str(args.learning_rate)+'_wd_' + str(args.weight_decay)+'_bs_' + str(args.batch_size)+'_optim_' + str(args.optim)
    if args.data_augment:
        outString = outString + "_transform"
    else:
        outString = outString + "_notransform"
    print(outString)

    ordering = np.random.permutation(args.orig_trainset_size)
    np.save(outString+"_ordering.npy", ordering)
    print("Saved ordering of data points while training to rely less on seed setting during scrubbing")
    selection = ordering[:args.used_training_size]

    train_dataset, val_dataset = getDatasets(name=args.dataset, val_also=True, include_indices=selection, exclude_indices=args.exclude_indices, data_augment=args.data_augment)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    
    print("Training set length: ", len(train_dataset))
    print("Validation set length: ", len(val_dataset))

    exec("from models import %s" % args.model)
    model = eval(args.model)().to(device)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    if device=='cuda':
        model = torch.nn.DataParallel(model)
    
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        error('unknown optimizer')


    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80,120], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, verbose=True)
    
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, epoch, args.epochs, optim=optim, device=device, outString=outString)

        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, epoch, args.epochs, optim=None, device=device, outString=outString)

        tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        #if val_accuracy > best_accuracy:
        #    print('Saving model...')
        #    best_accuracy = val_accuracy
        #    torch.save(model.state_dict(), 'trained_models/test.pt')

        lr_scheduler.step()          # For MultiStepLR
        # lr_scheduler.step()          # For CosineAnnealingLR
        # lr_scheduler.step(val_loss)  # For ReduceLROnPlateau 

    print('Saving model...')
    torch.save(model.state_dict(), outString + '.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network')
    arg_parser.add_argument('--train_seed', type=int, default=0)
    arg_parser.add_argument('--orig_trainset_size', type=int, default=50000)
    arg_parser.add_argument('--used_training_size', type=int, default=1000)
    arg_parser.add_argument('--dataset', type=str, default='cifar10')
    arg_parser.add_argument('--model', type=str, default='CIFAR10Net')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    arg_parser.add_argument('--epochs', type=int, default=200)
    arg_parser.add_argument('--n_classes', type=int, default=10)
    arg_parser.add_argument('--learning_rate', type=float, default=0.1)
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay, or l2_regularization for SGD')
    arg_parser.add_argument('--exclude_indices', type=list, default=None, help='list of indices to leave out of training.')
    arg_parser.add_argument('--data_augment', default=False, action='store_true')
    args = arg_parser.parse_args()
    main(args)
