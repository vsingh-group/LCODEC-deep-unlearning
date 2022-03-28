import argparse

import copy
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import getDatasets
from nn_utils import do_epoch

from scrub_tools import scrubSample, inp_perturb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
def main(args):
    outString = 'trained_models/'+args.dataset+"_"+args.model+"_run_" + str(args.run) + '_epochs_' + str(args.train_epochs)

    print('DEPRECATED, use multiscrub')

    tmp = {}
    tmp['selectionType'] = [args.selectionType]
    tmp['order'] = [args.order]
    tmp['HessType'] = [args.HessType]
    tmp['scrub_ID'] = [args.scrub_index]
    tmp['run'] = [args.run]
    tmp['orig_trainset_size'] = args.orig_trainset_size
    tmp['delta'] = args.delta
    tmp['epsilon'] = args.epsilon

    train_dataset, val_dataset = getDatasets(name=args.dataset, include_indices=[args.scrub_index])
    
    print('Validation Set Size: ', len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=1)

    exec("from models import %s" % args.model)
    model = eval(args.model)().to(device)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    model.load_state_dict(torch.load(args.MODEL_FILE))

    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)
        print(f'Before: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

    tmp['val_acc_before'] = [val_accuracy]
    tmp['val_loss_before'] = [val_loss]

    if args.scrubType == 'HC':
        updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = scrubSample(model, train_dataset, criterion, args, optim, device, outString=outString)
    elif args.scrubType == 'IP':
        updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = inp_perturb(model, train_dataset, criterion, args, optim, device, outString=outString)
    else:
        print('Unknown scrubType: ', args.scrubType)
        return NotImplementedError
    model.load_state_dict(updatedSD)


    tmp['sample_loss_before'] = samplossbefore.detach().cpu()
    tmp['sample_loss_after'] = samplossafter.detach().cpu()
    tmp['sample_gradnorm_before'] = gradnormbefore
    tmp['sample_gradnorm_after'] = gradnormafter
    #best_accuracy = 0
    #train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

    #model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)

    print(f'After: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

    print('Saving model...')
    updatedmodelname = args.MODEL_FILE.split('.')[0] + '_scrubbed_' + str(args.scrub_index) + '_' + args.selectionType + '_' + args.order + '.pt'
    torch.save(model.state_dict(), updatedmodelname)

    tmp['val_acc_after'] = [val_accuracy]
    tmp['val_loss_after'] = [val_loss]

    df = pd.DataFrame(tmp)
    #fname = f'res/results_{args.data}_{args.model}_{args.conf}_{args.run}.csv'
    if os.path.isfile(args.outfile):
        df.to_csv(args.outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(args.outfile, mode='a', header=True, index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Scrub a sample')
    arg_parser.add_argument('--dataset', type=str, default='cifar10')
    arg_parser.add_argument('--model', type=str, default='CIFAR10Net')
    arg_parser.add_argument('--MODEL_FILE', type=str, default="trained_models/full.pt", help='A model in trained_models, trained using train.py')
    arg_parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs model was originally trained for, used to get last two gradients')
    arg_parser.add_argument('--interp_size', type=int, default=32, help='Size of input image to interpolate for hypercolumns')
    arg_parser.add_argument('--scrub_index', type=int, default=None, help='Index of an example to scrub.')
    arg_parser.add_argument('--orig_trainset_size', type=int, default=None, help='size of orig training set')
    arg_parser.add_argument('--epsilon', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--delta', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--l2_reg', type=float, default=0.001, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    arg_parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    arg_parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    arg_parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    arg_parser.add_argument('--n_perturbations', type=int, default=50)
    arg_parser.add_argument('--order', type=str, default='BP', choices=['BP','Hessian'])
    arg_parser.add_argument('--selectionType', type=str, default='Full', choices=['Full', 'FOCI', 'Random'])
    arg_parser.add_argument('--run', type=int, default=1, help='Repitition index.')
    arg_parser.add_argument('--outfile', type=str, default="scrub_ablate_results.csv", help='output file name to append to')
    arg_parser.add_argument('--updatedmodelname', type=str, help='output file name to append to')
    args = arg_parser.parse_args()

    manual_seed(args.run)

    main(args)
