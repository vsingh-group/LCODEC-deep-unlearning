# KFAC tutorial: https://towardsdatascience.com/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414
# KFAC tutorial: https://yaroslavvb.medium.com/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
# KFAC main code: https://github.com/cybertronai/autograd-lib


import argparse
import pdb

import copy
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
from tqdm import tqdm

import time

from data_utils import getDatasets
from nn_utils import do_epoch, manual_seed, retrain_model

from scrub_tools import scrubSample, inp_perturb

from autograd_lib import util as u
from autograd_lib import autograd_lib
from collections import defaultdict

from attrdict import AttrDefault
from einops import rearrange
from grad_utils import gradNorm


def KFAC_scrub_step(model, residual_dataset, scrub_dataset, criterion, args, optim, device, outString):

    n_residual = len(residual_dataset)
    n_scrub = 1

    # getting Hesssian for residual set
    residual_activations = {}
    residual_KFAC_hessians = defaultdict(lambda: AttrDefault(float))

    residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=256,
                                                 shuffle=False, num_workers=1)

    for x, y_true in tqdm(residual_loader, leave=True):
        def save_activations(layer, A, _):
            residual_activations[layer] = A
            residual_KFAC_hessians[layer].AA += torch.einsum("ni,nj->ij", A, A)

        with autograd_lib.module_hook(save_activations):
            data_batch = x.to(device)
            targets_batch = y_true.to(device)
            Y = model(data_batch)
            loss = criterion(Y, targets_batch)

        def compute_hess(layer, _, B):
            residual_KFAC_hessians[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy', retain_graph=True)

    # getting Hessian for scrubbed set/ point
    model.zero_grad()
    optim.zero_grad()

    scrubee_activations = {}
    scrubee_KFAC_hessians = defaultdict(lambda: AttrDefault(float))

    for i in range(n_scrub):
        def save_activations(layer, A, _):
            scrubee_activations[layer] = A
            scrubee_KFAC_hessians[layer].AA += torch.einsum("ni,nj->ij", A, A)

        with autograd_lib.module_hook(save_activations):
            data_batch = scrub_dataset[i][0].to(device)
            targets_batch = torch.tensor([scrub_dataset[i][1]]).to(device)
            Y = model(data_batch)
            scrub_loss = criterion(Y, targets_batch)

        def compute_hess(layer, _, B):
            scrubee_KFAC_hessians[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy', retain_graph=True)

    #to generate gradients with respect to the scrubbed point
    scrub_loss.backward()
    fullprevgradnorm = gradNorm(model)

    num_layers = len(model.layers)
    print("Number of layers in MLP: ", num_layers)

    for i in range(num_layers):

        residual_hess_layer = residual_KFAC_hessians[model.layers[i]]
        residual_hess = torch.einsum('kl,ij->kilj', residual_hess_layer.BB / n_residual, residual_hess_layer.AA / n_residual)
        residual_hess = rearrange(residual_hess, 'd0 d1 d2 d3 -> (d0 d1) (d2 d3)')

        scrubee_hess_layer = scrubee_KFAC_hessians[model.layers[i]]
        scrubee_hess = torch.einsum('kl,ij->kilj', scrubee_hess_layer.BB / n_residual, scrubee_hess_layer.AA / n_residual)
        scrubee_hess = rearrange(scrubee_hess, 'd0 d1 d2 d3 -> (d0 d1) (d2 d3)')

        diff_hess = ((n_residual+1)*residual_hess - scrubee_hess)/n_residual
        smoothed_diff_hess = diff_hess + args.l2_reg*torch.eye(diff_hess.shape[0]).to(device)

        scrubee_gradient = model.layers[i].weight.grad
        layer_weights = model.layers[i].weight
        vector_weights = p2v(layer_weights)

        diff_H_inv_times_grad = torch.linalg.solve(smoothed_diff_hess, p2v(scrubee_gradient))
        
        vector_weights += (1/n_residual)*diff_H_inv_times_grad

        # This is weight, update not sure, to check??
        v2p(vector_weights, layer_weights)

    
    # Get loss and grad norm on updated model 
    data_batch = scrub_dataset[0][0].to(device)
    targets_batch = torch.tensor([scrub_dataset[0][1]]).to(device)
    Y = model(data_batch)
    scrub_loss_after = criterion(Y, targets_batch)
    optim.zero_grad()
    scrub_loss_after.backward()

    fullscrubbedgradnorm = gradNorm(model)

    return model.state_dict(), scrub_loss, scrub_loss_after, fullprevgradnorm, fullscrubbedgradnorm


def scrubMany(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.train_epochs)+'_lr_' + str(args.train_lr)+'_wd_' + str(args.train_wd)+'_bs_' + str(args.train_bs)+'_optim_' + str(args.train_optim)
    if args.data_augment:
        outString = outString + "_transform"
    else:
        outString = outString + "_notransform"
    
    # pdb.set_trace()
    model: u.SimpleMLP = u.SimpleMLP([784, 20, 10], nonlin=True, bias=False)
    model = model.to(device)
    autograd_lib.register(model)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_dataset, val_dataset = getDatasets(name=args.dataset, val_also=True, data_augment=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    
    print("Training set length: ", len(train_dataset))
    print("Validation set length: ", len(val_dataset))

    if device=='cuda':
        model = torch.nn.DataParallel(model)
    
    optim = torch.optim.SGD(model.parameters(), lr=args.train_lr, momentum=args.train_momentum, weight_decay=args.train_wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
    
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    if args.do_train:
        for epoch in range(args.train_epochs):
            train_loss, train_accuracy = do_epoch(model, train_loader, criterion, epoch, args.train_epochs, optim=optim, device=device, outString=outString)

            with torch.no_grad():
                val_loss, val_accuracy = do_epoch(model, val_loader, criterion, epoch, args.train_epochs, optim=None, device=device, outString=outString)

            tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                       f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

            lr_scheduler.step()          # For CosineAnnealingLR

        print('Saving model...')
        torch.save(model.state_dict(), outString + '.pt')
    else:
        # reload model to trained state
        model.load_state_dict(torch.load(outString+".pt"))

    tmp = {}
    tmp['dataset'] = [args.dataset]
    tmp['model'] = [args.model]
    tmp['train_epochs'] = [args.train_epochs]
    tmp['selectionType'] = [args.selectionType]
    tmp['order'] = [args.order]
    tmp['HessType'] = [args.HessType]
    tmp['approxType'] = [args.approxType]
    tmp['run'] = [args.run]
    tmp['cuda_id'] = [args.run]
    tmp['seed'] = [args.run]
    tmp['orig_trainset_size'] = [args.orig_trainset_size]
    tmp['used_training_size'] = [args.used_training_size]
    tmp['delta'] = [args.delta]
    tmp['epsilon'] = [args.epsilon]
    tmp['l2_reg'] = [args.l2_reg]

    if args.do_KFAC:
        tmp['KFAC_scrub'] = [1]
    else:
        tmp['KFAC_scrub'] = [0]

    ordering = np.random.permutation(args.orig_trainset_size)
    full_dataset, val_dataset = getDatasets(name=args.dataset, val_also=True, data_augment=False)
    scrubbed_list = []
        
    print('Validation Set Size: ', len(val_dataset))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=1)
    with torch.no_grad():
        val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)
        print(f'Model:{args.model} Before: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        tmp['val_acc_before'] = [val_accuracy]
        tmp['val_loss_before'] = [val_loss]

    prev_val_acc = val_accuracy

    i = 0
    j = 0
    while i < args.n_removals:
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))

        if args.scrub_batch_size is not None:

            # select samples to scrub
            scrub_list = []
            while len(scrub_list) < args.scrub_batch_size and (i+len(scrub_list)) < args.n_removals:
                scrubee = ordering[j]
                if full_dataset[scrubee][1] == args.removal_class:
                    scrub_list.append(scrubee)
                j += 1

        else:
            # randomly select samples to scrub
            #scrubee = ordering[j]
            scrub_list = [ordering[j]]
            j += 1

        #scrub_dataset = Subset(full_dataset, [scrubee])
        scrub_dataset = Subset(full_dataset, scrub_list)

        #scrubbed_list.append(scrubee)
        scrubbed_list.extend(scrub_list)

        residual_dataset, _ = getDatasets(name=args.dataset, val_also=False, exclude_indices=scrubbed_list)
        residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=1)
        
        print('Residual dataset size: ', len(residual_dataset))
        #print('Removing: ', i, scrubee)
        print('Removing: ', i, scrub_list)

        #tmp['scrubee'] = [scrubee]
        tmp['scrub_list'] = [scrub_list]
        tmp['n_removals'] = [i]

        prev_statedict_fname = outString + '_prevSD.pt'
        torch.save(model.state_dict(), prev_statedict_fname)

        # because we reload the model
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)

        if args.do_KFAC:
            print("Runnning KFAC based scrubbing")
            updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = KFAC_scrub_step(model, residual_dataset, scrub_dataset, criterion, args, optim, device, outString=outString)
        else:
            print("Runing FOCI based scrubbing")
            foci_val, updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = inp_perturb(model, scrub_dataset, criterion, args, optim, device, outString=outString)

        # reload for deepcopy
        # apply new weights
        # without this cannot deepcopy later
        model: u.SimpleMLP = u.SimpleMLP([784, 20, 10], nonlin=True, bias=False)
        model = model.to(device)
        autograd_lib.register(model)
        model.load_state_dict(updatedSD)
        
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)

        #print(f'After: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        print(f'\t Previous Val Acc: {prev_val_acc}')
        print(f'\t New Val Acc: {val_accuracy}')


        if prev_val_acc - val_accuracy > args.val_gap_skip:
            print('########## BAD SAMPLE BATCH DETECTED, REVERTING MODEL #######')
            model: u.SimpleMLP = u.SimpleMLP([784, 20, 10], nonlin=True, bias=False)
            model = model.to(device)
            autograd_lib.register(model)
            model.load_state_dict(torch.load(prev_statedict_fname))
            tmp['bad_sample'] = 1
        else:
            prev_val_acc = val_accuracy
            tmp['bad_sample'] = 0
            i += len(scrub_list)

        tmp['time'] = time.time()

        tmp['val_acc_after'] = [val_accuracy]
        tmp['val_loss_after'] = [val_loss]

        tmp['sample_loss_before'] = [samplossbefore.detach().cpu().item()]
        tmp['sample_loss_after'] = [samplossafter.detach().cpu().item()]
        tmp['sample_gradnorm_before'] = [gradnormbefore]
        tmp['sample_gradnorm_after'] = [gradnormafter]

        resid_loss, resid_accuracy, resid_gradnorm = do_epoch(model, residual_loader, criterion, 0, 0, optim=None, device=device, compute_grads=True)
        print('Residual Gradnorm:', resid_gradnorm)
        tmp['residual_loss_after'] = [resid_loss]
        tmp['residual_acc_after'] = [resid_accuracy]
        tmp['residual_gradnorm_after'] = [resid_gradnorm]


        df = pd.DataFrame(tmp)
        if os.path.isfile(args.outfile):
            df.to_csv(args.outfile, mode='a', header=False, index=False)
        else:
            df.to_csv(args.outfile, mode='a', header=True, index=False)

            
    return model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Scrub a sample')
    arg_parser.add_argument('--dataset', type=str, default='mnist')
    arg_parser.add_argument('--model', type=str, default='SimpleMLP')
    arg_parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs model was originally trained for, used to get last two gradients')
    arg_parser.add_argument('--interp_size', type=int, default=32, help='Size of input image to interpolate for hypercolumns')
    arg_parser.add_argument('--n_removals', type=int, default=1000, help='number of samples to scrub')
    arg_parser.add_argument('--orig_trainset_size', type=int, default=60000, help='size of orig training set')
    arg_parser.add_argument('--used_training_size', type=int, default=60000, help='number of data points actually used for training')
    arg_parser.add_argument('--epsilon', type=float, default=0.1, help='scrubbing rate')
    arg_parser.add_argument('--delta', type=float, default=0.01, help='scrubbing rate')
    arg_parser.add_argument('--l2_reg', type=float, default=0.01, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    arg_parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    arg_parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    arg_parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    arg_parser.add_argument('--n_perturbations', type=int, default=1000)
    arg_parser.add_argument('--order', type=str, default='Hessian', choices=['BP','Hessian'])
    arg_parser.add_argument('--selectionType', type=str, default='FOCI', choices=['Full', 'FOCI', 'Random', 'One'])
    arg_parser.add_argument('--FOCIType', type=str, default='full', choices=['full','cheap'])
    arg_parser.add_argument('--run', type=int, default=1, help='Repitition index / CUDA ID / Seed')
    arg_parser.add_argument('--outfile', type=str, default="FOCI_scrub_results.csv", help='output file name to append to')
    arg_parser.add_argument('--updatedmodelname', type=str, help='output file name to append to')
    arg_parser.add_argument('--hessian_device', type=str, default='cpu', help='Device for Hessian computation')
    arg_parser.add_argument('--val_gap_skip', type=float, default=0.05, help='validation drop for skipping a sample to remove (should retrain)')
    arg_parser.add_argument('--scrub_batch_size', type=int, default=None)
    arg_parser.add_argument('--removal_class', type=int, default=0)
    
    # Added for new outstring
    arg_parser.add_argument('--do_train', default=False, action='store_true')
    arg_parser.add_argument('--train_lr', type=float, default=0.0001, help="training learning rate")
    arg_parser.add_argument('--train_wd', type=float, default=0.01, help="training weight decay")
    arg_parser.add_argument('--train_momentum', type=float, default=0.9, help="training momentum for SGD")
    arg_parser.add_argument('--train_bs', type=int, default=128, help="training batch size")
    arg_parser.add_argument('--train_optim', type=str, default='sgd', choices=['sgd', 'adam'], help="training optimizer")
    arg_parser.add_argument('--data_augment', type=int, default=0, help='whether to augment or not') 

    # Arugment to swith between KFAC and FOCI based scrubbing
    arg_parser.add_argument('--do_KFAC', default=False, action='store_true')

    args = arg_parser.parse_args()

    manual_seed(args.run)

    scrubMany(args)


