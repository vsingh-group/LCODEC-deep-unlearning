import argparse

import copy
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import time

from data_utils import getDatasets
from nn_utils import do_epoch, manual_seed

from sklearn.metrics import classification_report

from vgg_scrub_tools import vgg_inp_perturb

from VGGFaceDataset import getVGGClassLabelFromName, VGGFaceDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scrubMany(args):

    prevString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.train_epochs)
    outString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.train_epochs)+'_eps_'+str(args.epsilon)
    
    tmp = {}
    tmp['dataset'] = [args.dataset]
    tmp['model'] = [args.model]
    tmp['train_epochs'] = [args.train_epochs]
    tmp['selectionType'] = [args.selectionType]
    tmp['order'] = [args.order]
    tmp['HessType'] = [args.HessType]
    tmp['approxType'] = [args.approxType]
    tmp['run'] = [args.run]
    tmp['orig_trainset_size'] = [args.orig_trainset_size]
    tmp['delta'] = [args.delta]
    tmp['epsilon'] = [args.epsilon]
    tmp['l2_reg'] = [args.l2_reg]


    exec("from models import %s" % args.model)
    model = eval(args.model)().to(device).float()#.double()
    model_copy = eval(args.model)().to(args.sec_device).float()#.double()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # reload model to trained state
    model.load_state_dict(torch.load(args.MODEL_FILE))
    model_copy.load_state_dict(torch.load(args.MODEL_FILE))

    criterion = torch.nn.CrossEntropyLoss()

    classIDtoScrub = getVGGClassLabelFromName(args.id_to_scrub)

    scrub_dataset = VGGFaceDataset(attr_data_file='100_dled_vggface.csv', person=args.id_to_scrub, exclude=False)
    residual_dataset = VGGFaceDataset(attr_data_file='100_dled_vggface.csv', person=args.id_to_scrub, exclude=True)
    scrubbed_list = []
        
    ordering = np.random.permutation(len(scrub_dataset))

    scrubbed_loader = torch.utils.data.DataLoader(scrub_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=1)
    residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=1)
            
    print('Residual dataset size: ', len(residual_dataset))


    prev_scrubbed_gradnorm = 0

    i = 0
    j = 0
    while i < args.n_removals:
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))
        print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=args.sec_device)/1024./1024.))

        # randomly select samples to scrub
        #scrubee = ordering[j]
        scrub_list = [ordering[j]]
        j += 1

        #scrub_dataset = Subset(full_dataset, [scrubee])
        scrubee_dataset = Subset(scrub_dataset, scrub_list)

        #scrubbed_list.append(scrubee)
        scrubbed_list.extend(scrub_list)

        #print('Removing: ', i, scrubee)
        print('Removing: ', i, scrub_list)

        #tmp['scrubee'] = [scrubee]
        tmp['scrub_list'] = [scrub_list]
        tmp['n_removals'] = [i]

        # loops once for now, maybe more in future
        foci_val = 1
        while foci_val > args.cheap_foci_thresh:

            if i % 10 == 10:
                tmp['resid_calc'] = 1
                resid_loss, resid_accuracy = do_epoch(model, residual_loader, criterion, 0, 0, optim=None, device=device, compute_grads=False)
                print('\tResidual Loss:', resid_loss)
                print('\tResidual Accuracy:', resid_accuracy)
                #print('\tResidual Gradnorm:', resid_gradnorm)

                tmp['residual_loss'] = [resid_loss]
                tmp['residual_acc'] = [resid_accuracy]
                #tmp['residual_gradnorm'] = [resid_gradnorm]
            else:
                tmp['resid_calc'] = 0

            prev_statedict_fname = outString + '_prevSD.pt'
            torch.save(model.state_dict(), prev_statedict_fname)

            # because we reload the model
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)

            foci_val, updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = vgg_inp_perturb(model, model_copy, scrubee_dataset, criterion, args, optim, device, outString=prevString)
            foci_val = 0

            #import pdb; pdb.set_trace()
            torch.cuda.empty_cache()

            model.load_state_dict(updatedSD)
            model_copy.load_state_dict(updatedSD)

            torch.cuda.empty_cache()

            tmp['time'] = time.time()

            
            scrubbed_loss, scrubbed_accuracy, scrubbed_gradnorm = do_epoch(model, scrubbed_loader, criterion, 0, 0, optim=None, device=device, compute_grads=True)
            print('scrubbed Gradnorm:', scrubbed_gradnorm)
            print('scrubbed loss:', scrubbed_loss)
            print('scrubbed accuracy:', scrubbed_accuracy)
            tmp['scrubbed_acc_after'] = [scrubbed_accuracy]
            tmp['scrubbed_loss_after'] = [scrubbed_loss]
            tmp['scrubbed_gradnorm_after'] = [scrubbed_gradnorm]
            tmp['scrubbed_gradnorm_change'] = [scrubbed_gradnorm-prev_scrubbed_gradnorm]
            prev_scrubbed_gradnorm = scrubbed_gradnorm


            tmp['sample_loss_before'] = [samplossbefore]
            tmp['sample_loss_after'] = [samplossafter]
            tmp['sample_gradnorm_before'] = [gradnormbefore]
            tmp['sample_gradnorm_after'] = [gradnormafter]
            tmp['sample_gradnorm_change'] = [gradnormafter-gradnormbefore]

            df = pd.DataFrame(tmp)
            if os.path.isfile(args.outfile):
                df.to_csv(args.outfile, mode='a', header=False, index=False)
            else:
                df.to_csv(args.outfile, mode='a', header=True, index=False)

            i += len(scrub_list)

    return model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Scrub a sample')
    arg_parser.add_argument('--dataset', type=str, default='vgg')
    arg_parser.add_argument('--model', type=str, default='resnet')
    arg_parser.add_argument('--MODEL_FILE', type=str, default="trained_models/full.pt", help='A model in trained_models, trained using train.py')
    arg_parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs model was originally trained for, used to get last two gradients')
    arg_parser.add_argument('--interp_size', type=int, default=32, help='Size of input image to interpolate for hypercolumns')
    arg_parser.add_argument('--n_removals', type=int, default=10000, help='number of samples to scrub')
    arg_parser.add_argument('--orig_trainset_size', type=int, default=None, help='size of orig training set')
    arg_parser.add_argument('--epsilon', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--delta', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--l2_reg', type=float, default=0.001, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    arg_parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    arg_parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    arg_parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    arg_parser.add_argument('--n_perturbations', type=int, default=50)
    arg_parser.add_argument('--order', type=str, default='BP', choices=['BP','Hessian'])
    arg_parser.add_argument('--selectionType', type=str, default='Full', choices=['Full', 'FOCI', 'Random'])
    arg_parser.add_argument('--FOCIType', type=str, default='full', choices=['full','cheap'])
    arg_parser.add_argument('--cheap_foci_thresh', type=float, default=0.05, help='threshold for codec2 calls in cheap_foci')
    arg_parser.add_argument('--run', type=int, default=1, help='Repitition index.')
    arg_parser.add_argument('--outfile', type=str, default="scrub_ablate_results.csv", help='output file name to append to')
    arg_parser.add_argument('--updatedmodelname', type=str, help='output file name to append to')
    arg_parser.add_argument('--hessian_device', type=str, default='cpu', help='Device for Hessian computation')
    arg_parser.add_argument('--sec_device', type=str, default='cuda:1', help='Device for Second Model Copy computation')

    arg_parser.add_argument('--val_gap_skip', type=float, default=0.05, help='validation drop for skipping a sample to remove (should retrain)')
    arg_parser.add_argument('--scrub_batch_size', type=int, default=None)

    arg_parser.add_argument('--id_to_scrub', type=str, default='Aamir_Khan', help='Index of an example to scrub.')

    args = arg_parser.parse_args()

    manual_seed(args.run)

    scrubMany(args)
