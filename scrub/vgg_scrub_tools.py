import argparse
import random
import copy
import numpy as np
import pandas as pd
import os
import torch
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from hypercolumn import HyperC, ActivationsHook, NLP_ActivationsHook
from grad_utils import getGradObjs, gradNorm, getHessian, getVectorizedGrad, getOldPandG

import sys
sys.path.append('..')
#from codec import foci, cheap_foci
from codec import torch_foci as foci
from codec import cheap_torch_foci as cheap_foci

from scrub_tools import DisableBatchNorm
from scrub_tools import myFOCI, reverseLinearIndexingToLayers, updateModelParams
from scrub_tools import NewtonScrubStep, CR_NaiveNewton, NoisyReturn

def vgg_inp_perturb(model, model_copy, dataset, criterion, params, optim, device, outString, is_nlp=False):
    '''
        Works for slice selection of Linear (columns) and Convolution (filters) layers. 
    '''

    #print('prior lbfgs')
    #stats(model)
    #LBFGSTorch(model, datapoint, criterion, params, device)
    #print('post lbfgs')
    #stats(model)

    if is_nlp:
        num_tokens = dataset[0][0].shape[0]
        num_points = len(dataset)
        print("Number of tokens: ", num_tokens)
        print("Number of datapoints: ", num_points)

    if is_nlp:
        if params.scrub_batch_size is not None:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.scrub_batch_size,
                                                     shuffle=False, num_workers=1)
            nlp_batch = next(iter(data_loader))
            nlp_batch = tuple(t.to(device) for t in nlp_batch)
            nlp_inputs = {
                'input_ids': nlp_batch[0],
                'attention_mask': nlp_batch[1],
                #'token_type_ids': batch[2],
                'labels': nlp_batch[3]
            }
        else:
            nlp_inputs = {
                    'input_ids': dataset[0][0].to(device),
                    'attention_mask': dataset[0][1].to(device),
                    #'token_type_ids': dataset[0][2],
                    'labels': dataset[0][3].to(device)
                }
    
    else:
        if params.scrub_batch_size is not None:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.scrub_batch_size,
                                                     shuffle=False, num_workers=1)
            x, y_true = next(iter(data_loader))
            x = x.to(device)
            y_true = y_true.to(device)
        else:
            x, y_true = dataset[0][0], dataset[0][1]
            x = x.to(device)
            y_true = torch.Tensor([y_true]).type(torch.long).to(device)
            x.unsqueeze_(0)

    print(x.shape)
    print(y_true.shape)

    if is_nlp:
        myActs = NLP_ActivationsHook(model)
    else:
        myActs = ActivationsHook(model)

    torchLayers = myActs.getLayers()

    if is_nlp:
        myActs_copy = NLP_ActivationsHook(model_copy)
    else:
        myActs_copy = ActivationsHook(model_copy)

    torchLayers_copy = myActs_copy.getLayers()



    activations = []
    layers = None # same for all, filled in by loop below
    losses = []

    model.eval()
    for m in range(params.n_perturbations):

        if is_nlp:
            if params.scrub_batch_size is not None:

                tmp_x = nlp_inputs['input_ids'].clone()
                tokens_to_change = []
                for point in range(0, num_points):
                    attention_span = torch.count_nonzero(nlp_inputs['attention_mask'][point])
                    tokens_to_change.append(random.randint(1, attention_span-1))                             # change a token in the attention part of the sentence

                replacement_tokens = [random.randint(999, 30521) for point in range(0, num_points)]             # 999 to 30521 are valid tokens for the distilbert tokenizer

                for point in range(num_points):
                    tmp_x[point][tokens_to_change[point]] = replacement_tokens[point]

                tmpdata = {
                    'input_ids': tmp_x.to(device),
                    'attention_mask': nlp_inputs['attention_mask'],
                    #'token_type_ids': nlp_inputs['token_type_ids'] ,
                    'labels': nlp_inputs['labels']
                }

                acts, torchLayers, out = myActs.get_NLP_Activations(tmpdata)
                loss = criterion(out, tmpdata['labels'])
                vec_acts = p2v(acts)

            else:
                attention_span = torch.count_nonzero(nlp_inputs['attention_mask'])
                token_to_change = random.randint(1, attention_span-1)
                tmp_x = nlp_inputs['input_ids'].clone()
                tmp_x[token_to_change] = random.randint(999, 30521)  # 999 to 30521 are valid tokens for the distilbert tokenizer

                tmpdata = {
                    'input_ids': torch.unsqueeze(tmp_x,0).to(device),
                    'attention_mask': torch.unsqueeze(dataset[0][1],0).to(device),
                    #'token_type_ids': dataset[0][2],
                    'labels': torch.unsqueeze(dataset[0][3],0).to(device)
                }
                acts, out = myActs.get_NLP_Activations(tmpdata)
                loss = criterion(out, tmpdata['labels'])
                vec_acts = p2v(acts)

        else:
            tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
            acts, out = myActs.getActivations(tmpdata.to(device))
            loss = criterion(out, y_true)
            vec_acts = p2v(acts)

        activations.append(vec_acts.detach())
        losses.append(loss.detach())

    acts = torch.vstack(activations)
    losses = torch.Tensor(losses).to(device)

    # descructor is not called on return for this
    # call it manually
    myActs.clearHooks()
    myActs_copy.clearHooks()

    # run selection
    if params.selectionType == 'Full':
        selectedActs = np.arange(len(vec_acts)).tolist()

    elif params.selectionType == 'Random':
        foci_result, _ = foci(acts, losses, earlyStop=True, verbose=False)
        selectedActs = np.random.permutation(len(vec_acts))[:int(len(foci_result))]
        
    elif params.selectionType == 'One':
        selectedActs = [np.random.permutation(len(vec_acts))[0]]

    elif params.selectionType == 'FOCI':
        print('Running FOCI...')
        if params.FOCIType == 'full':
            print('Running full FOCI...')
            selectedActs, scores = foci(acts, losses, earlyStop=True, verbose=False)
        elif params.FOCIType == 'cheap':
            print('Running cheap FOCI...')
            selectedActs, scores = cheap_foci(acts, losses)
        else:
            error('unknown foci type')

    else: 
        error('unknown scrub type')

    # create mask for update
    # params_mask = [1 if i in params else 0 for i in range(vec_acts.shape[1])]

    import pdb; pdb.set_trace()

    slices_to_update = reverseLinearIndexingToLayers(selectedActs, torchLayers)
    slices_to_update_copy = reverseLinearIndexingToLayers(selectedActs, torchLayers_copy)
    print('Selected model blocks to update:')
    print(slices_to_update)

    ############ Sample Forward Pass ########
    model.train()
    model = DisableBatchNorm(model)
    total_loss = 0
    total_accuracy = 0

    if is_nlp:

        if params.scrub_batch_size is not None:
            x = {
                    'input_ids': nlp_inputs['input_ids'],
                    'attention_mask': nlp_inputs['attention_mask'],
                    #'token_type_ids': nlp_inputs['token_type_ids'],
                    'labels': nlp_inputs['labels']
                }
            y_true = x['labels']
            y_pred = model(**x)
            sample_loss_before = criterion(y_pred, y_true)

        else:
            x = {
                    'input_ids': torch.unsqueeze(dataset[0][0],0).to(device),
                    'attention_mask': torch.unsqueeze(dataset[0][1],0).to(device),
                    #'token_type_ids': datapoint[0][2],
                    'labels': torch.unsqueeze(dataset[0][3],0).to(device)
                }
            y_true = x['labels']
            y_pred = model(**x)
            sample_loss_before = criterion(y_pred, y_true)
    else:  
        y_pred = model(x)
        sample_loss_before = criterion(y_pred, y_true)
    print('Sample Loss Before: ', sample_loss_before)

    ####### Sample Gradient
    optim.zero_grad()
    sample_loss_before.backward()

    fullprevgradnorm = gradNorm(model)
    print('Sample Gradnorm Before: ', fullprevgradnorm)

    sampGrad1, _ = getGradObjs(model)
    vectGrad1, vectParams1, reverseIdxDict = getVectorizedGrad(sampGrad1, slices_to_update, device)
    #vectGrad1full = p2v(sampGrad1)
    #vectGrad1 = [vectGrad1full[i] for i in params]
    model.zero_grad()

    if params.order == 'Hessian':

        # old hessian
        #second_last_name = outString + '_epoch_'  + str(params.train_epochs-2) + "_grads.pt"
        #dwtlist = torch.load(second_last_name)
        #delwt, vectPOld, _ = getVectorizedGrad(dwtlist, slices_to_update, device)
        delwt, vectPOld = getOldPandG(outString, params.train_epochs-2, slices_to_update, device)

        #one_last_name = outString + '_epoch_'  + str(params.train_epochs-1) + "_grads.pt"
        #dwtm1list = torch.load(one_last_name)
        #delwtm1, vectPOld_1, _ = getVectorizedGrad(dwtm1list, slices_to_update, device)
        delwtm1, vectPOld_1 = getOldPandG(outString, params.train_epochs-1, slices_to_update, device)

        print ('before old hessian GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=params.sec_device)/1024./1024.))

        oldHessian = getHessian(delwt, delwtm1, params.approxType, w1=vectPOld, w2=vectPOld_1, hessian_device=params.hessian_device)
        print ('after old hessian GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=params.sec_device)/1024./1024.))

        # sample hessian
        model_copy.train()
        model_copy = DisableBatchNorm(model_copy)

        # for finite diff use a small learning rate
        # default adam is 0.001/1e-3, so use it here
        optim_copy = torch.optim.SGD(model_copy.parameters(), lr=1e-3)

        if is_nlp:
            y_pred = model_copy(**x)
        else:    
            y_pred = model_copy(x.to(params.sec_device))
        loss = criterion(y_pred, y_true.to(params.sec_device))
        optim_copy.zero_grad()
        loss.backward()

        # step to get model at next point, compute gradients
        optim_copy.step()

        if is_nlp:
            y_pred = model_copy(**x)
        else:    
            y_pred = model_copy(x.to(params.sec_device))
        loss = criterion(y_pred, y_true.to(params.sec_device))
        optim_copy.zero_grad()
        loss.backward()

        print('Sample Loss after Step for Hessian: ', loss)

        sampGrad2, _ = getGradObjs(model_copy)
        vectGrad2, vectParams2, _ = getVectorizedGrad(sampGrad2, slices_to_update_copy, params.sec_device)

        print ('before sample hessian GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=params.sec_device)/1024./1024.))

        sampleHessian = getHessian(vectGrad1, vectGrad2, params.approxType, w1=vectParams1, w2=vectParams2, hessian_device=params.hessian_device)

        if params.HessType == 'Sekhari':
            # Sekhari unlearning update
            n = params.orig_trainset_size
            combinedHessian = (1/(n-1))*(n*oldHessian.to(params.hessian_device) - sampleHessian.to(params.hessian_device))

            updatedParams = NewtonScrubStep(vectParams1, vectGrad1, combinedHessian, n, l2lambda=params.l2_reg, hessian_device=params.hessian_device)
            updatedParams = NoisyReturn(updatedParams, nsamps=n, m=1, lamb=params.l2_reg, epsilon=params.epsilon, delta=params.delta, device=device)

        elif params.HessType == 'CR':
            updatedParams = CR_NaiveNewton(vectParams1, vectGrad1, sampleHessian, l2lambda=params.l2_reg, hessian_device=params.hessian_device)

        else:
            error('Unknown Hessian Update Type.')

    elif params.order == 'BP':
        updatedParams = vectParams1 + params.lr*vectGrad1

    else:
        error('unknown scrubtype')

    with torch.no_grad():
        updateModelParams(updatedParams, reverseIdxDict, model)

    if is_nlp:
        y_pred = model(**x)
    else:    
        y_pred = model(x)
    loss2 = criterion(y_pred, y_true)
    print('Sample Loss After: ', loss2)
    optim.zero_grad()
    loss2.backward()

    fullscrubbedgradnorm = gradNorm(model)
    print('Sample Gradnorm After: ', fullscrubbedgradnorm)

    model.zero_grad()

    # for future multiple scrubbing for a single sample
    #if params.FOCIType == 'cheap':
    #    foci_val = scores[0]
    #else:
    #    foci_val = 1
    foci_val = 0

    return foci_val, model.state_dict(), sample_loss_before, loss2, fullprevgradnorm, fullscrubbedgradnorm

