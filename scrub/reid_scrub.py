import torchreid
import pdb


import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


from tqdm import tqdm, trange
import numpy as np

import copy
import pandas as pd
import os
import sys
from torch.utils.data import Subset
from tqdm import tqdm

import time

from data_utils import getDatasets
from nn_utils import manual_seed

from scrub_tools import scrubSample, inp_perturb, reid_inp_perturb, DisableBatchNorm
from grad_utils import getGradObjs, gradNorm
from data_utils import SubsetDataWrapper

from reid_viz import visactmap

def compute_residual_gradnorm(model, residual_loader, engine, device):
    total_gradnorm = 0

    for batch in tqdm(residual_loader, desc="Residual Norm Evaluation"):

        imgs = batch['img'].to(device)
        pids = batch['pid'].to(device)

        logits = model(imgs)
        loss = engine.compute_loss(engine.criterion, logits, pids)

        model.zero_grad()
        loss.backward()
        batch_norm = gradNorm(model)

        total_gradnorm += batch_norm

    return total_gradnorm/len(residual_loader)


def main():

    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)
    manual_seed(seed=args.seed)

    original_stdout = sys.stdout

    main_folder = "./log/"+args.dataset+"_"+args.model_name

    output_file = main_folder+args.outfile

    score_file = open(main_folder+'/rem'+str(args.removal_class)+'_scores_ep'+str(args.epsilon)+'.txt', 'w')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=args.dataset,
        targets=args.dataset,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=512,
        transforms=['random_flip', 'random_crop'],
        combineall=True
    )


    model = torchreid.models.build_model(
        name=args.model_name,
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )



    if args.mode not in {'test', 'train', 'scrub'}:
        raise ValueError(f"unknown mode {args.mode}, use 'test' or 'train' or 'scrub'")


    if args.mode == 'train':

        print('start training')
            

    elif args.mode == "scrub":
        print("************Scrubbing**************")
        print('loading model')

        model_path = main_folder+"/model/model.pth.tar-"+str(args.train_epochs)
        torchreid.utils.load_pretrained_weights(model, model_path)
        model = model.cuda()

        print("Loading training data")
        train_data = datamanager.train_set
        vis_test_loader = datamanager.test_loader

        # outString = 'trained_models/'+"Ledgar"+"_"+"distilbert"+"_run_" + str(args.seed) + '_epochs_' + str(args.train_epochs)
    
        tmp = {}
        tmp['class_removed'] = [args.removal_class]
        tmp['train_epochs'] = [args.train_epochs]
        tmp['selectionType'] = [args.selectionType]
        tmp['order'] = [args.order]
        tmp['HessType'] = [args.HessType]
        tmp['approxType'] = [args.approxType]
        tmp['fociType'] = [args.FOCIType]
        tmp['num_perturb'] = [args.n_perturbations]
        tmp['run'] = [args.seed] 
        args.orig_trainset_size = len(train_data)
        tmp['orig_trainset_size'] = [len(train_data)]
        tmp['delta'] = [args.delta]
        tmp['epsilon'] = [args.epsilon]
        tmp['l2_reg'] = [args.l2_reg]

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


        scrubbed_list = []
        num_scrubbed = 0
        ordering = np.random.permutation(args.orig_trainset_size)
        j = 0
        
        print("@@@@@@@@@@@@@@@@@@@@@ For Inital  loaded model @@@@@@@@@@@@@@@@@@@@@")

        optimizer = torchreid.optim.build_optimizer(
            model,
            optim='adam',
            lr=0.0003
        )

        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )

        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smooth=True
        )
        use_gpu = torch.cuda.is_available()
        vis_dir = main_folder+"/eps_"+str(args.epsilon)+"_nscrub_"+str(num_scrubbed)
        visactmap(args.removal_class, model, vis_test_loader, vis_dir, 128, 256, use_gpu)

        sys.stdout = score_file
        engine.run(
            save_dir=main_folder,
            max_epoch=10,
            eval_freq=10,
            print_freq=1,
            test_only=True
        )
        print("@@@@@@@@@@@@@@@@@@@@@ For Inital  loaded model @@@@@@@@@@@@@@@@@@@@@")
        sys.stdout = original_stdout

        # for i in range(args.n_removals):
        while num_scrubbed < args.n_removals:

            if j>=args.orig_trainset_size:
                print("Removed all samples pertaining to the class ", args.removal_class)
                pdb.set_trace()

            if args.scrub_batch_size is not None:

                # select batch of samples to scrub

                scrub_list = []
                while len(scrub_list) < args.scrub_batch_size and (num_scrubbed+len(scrub_list)) < args.n_removals:
                    scrubee = ordering[j]
                    scrubee_class = train_data[scrubee]['pid']
                    if scrubee_class == args.removal_class:
                        scrub_list.append(scrubee)
                    j += 1

            else:

                # select sample to scrub

                scrubee_class = -1
                while scrubee_class != args.removal_class:
                    scrubee = ordering[j]
                    scrubee_class = train_data[scrubee]['pid']
                    j += 1

                scrub_list = [scrubee]
            
            num_scrubbed += len(scrub_list)
            scrub_dataset = Subset(train_data, scrub_list)
            scrubbed_list.extend(scrub_list)

            residual_dataset = SubsetDataWrapper(train_data, include_indices=None, exclude_indices=scrubbed_list)
            residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=1)
            
            print('Residual dataset size: ', len(residual_dataset))
            print('Removing: ', num_scrubbed, scrub_list, scrubee_class)

            tmp['scrub_list'] = [scrub_list]
            tmp['n_removals'] = [num_scrubbed]
            updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = reid_inp_perturb(model, scrub_dataset, engine, args, device, main_folder+"/vaults_"+engine.datamanager.sources[0])

            # reload for deepcopy
            # apply new weights
            print('reloading model for applying new weights')
            model = torchreid.models.build_model(
                name=args.model_name,
                num_classes=datamanager.num_train_pids,
                loss='softmax',
                pretrained=False
            )
            model.to(device)
            model.load_state_dict(updatedSD)
            model = DisableBatchNorm(model)

            # because we reload the model
            optimizer = torchreid.optim.build_optimizer(
                model,
                optim='adam',
                lr=0.0003
            )

            scheduler = torchreid.optim.build_lr_scheduler(
                optimizer,
                lr_scheduler='single_step',
                stepsize=20
            )

            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                label_smooth=True
            )

            tmp['sample_loss_before'] = [samplossbefore.detach().cpu().item()]
            tmp['sample_loss_after'] = [samplossafter.detach().cpu().item()]
            tmp['sample_gradnorm_before'] = [gradnormbefore]
            tmp['sample_gradnorm_after'] = [gradnormafter]

            resid_gradnorm = compute_residual_gradnorm(model, residual_loader, engine, device)
            tmp['residual_gradnorm_after'] = [resid_gradnorm]

            tmp['time'] = time.time()

            with torch.no_grad():
                vis_dir = main_folder+"/eps_"+str(args.epsilon)+"_nscrub_"+str(num_scrubbed)
                visactmap(args.removal_class, model, vis_test_loader, vis_dir, 128, 256, use_gpu)
                sys.stdout = score_file
                print("@@@@@@@@@@@@@@@@@@@@@ For Reloaded & Updated model @@@@@@@@@@@@@@@@@@@@@")
                engine.run(
                    save_dir=main_folder,
                    max_epoch=10,
                    eval_freq=10,
                    print_freq=1,
                    test_only=True
                )
                print("@@@@@@@@@@@@@@@@@@@@@ For Reloaded & Updated model @@@@@@@@@@@@@@@@@@@@@")
                sys.stdout = original_stdout

            df = pd.DataFrame(tmp)
            if os.path.isfile(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, mode='a', header=True, index=False)
            
        # print("Saving removed data_points")
        # np.save("results/NLP/NLP_scrubbed_class_"+str(args.removal_class)+"_foci_"+args.FOCIType+"_flips_"+str(args.n_tokens_to_replace)+"_epsilon_"+str(args.epsilon)+".npy", np.asarray(scrubbed_list))

    else:
        print("Test mode")




def build_arg_parser():
    parser = argparse.ArgumentParser()
    # Original arguments from distilbert_baseline file
    parser.add_argument("--mode", default="scrub", type=str, required=True, help="which mode: 'train' or 'test' or 'scrub'")
    parser.add_argument("--seed", default=0xDEADBEEF, type=int, required=False, help="seed for random number generation, default 0xDEADBEEF")

    parser.add_argument("--batch_size", default=32, type=int, required=False, help="training batch size, default 32")
    parser.add_argument("--train_epochs", default=10, type=int, required=False, help="number of epochs of training, default 10")
    
    # parser.add_argument("--weight_decay", default=0.01, type=float, required=False, help="AdamW weight decay, default 0.01")
    
    parser.add_argument("--learning_rate", default=0.0003, type=float, required=False, help="AdamW learning rate, default 5e-5")
    
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False, help="AdamW epsilon, default 1e-8")
    # parser.add_argument("--warmup_steps", default=0, type=int, required=False, help="Warmup steps for learning rate schedule, default 0")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False, help="max norm for gradient clipping, default 1.0")


    # Arguments from multi scrub
    parser.add_argument('--dataset', type=str, default=None, required=True)
    parser.add_argument('--model_name', type=str, default=None, required=True)

    parser.add_argument('--n_removals', type=int, default=100, help='number of samples to scrub')
    parser.add_argument('--removal_class', type=int, default=None, required=True, help='id of class to be removed')
    parser.add_argument('--orig_trainset_size', type=int, default=29419, help='size of orig training set')
    parser.add_argument('--epsilon', type=float, default=0.0005, help='scrubbing rate')
    parser.add_argument('--delta', type=float, default=0.01, help='scrubbing rate')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    
    parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    
    parser.add_argument('--n_perturbations', type=int, default=1000)
    parser.add_argument('--order', type=str, default='Hessian', choices=['BP','Hessian'])
    parser.add_argument('--selectionType', type=str, default='FOCI', choices=['Full', 'FOCI', 'Random'])
    parser.add_argument('--FOCIType', type=str, default='full', choices=['full','cheap'])
    
    parser.add_argument('--cheap_foci_thresh', type=float, default=0.05, help='threshold for codec2 calls in cheap_foci')
    parser.add_argument('--outfile', type=str, default="/person_reid_srub.csv", help='output file name to append to')
    parser.add_argument('--hessian_device', type=str, default='cpu', help='Device for Hessian computation')
    parser.add_argument('--scrub_batch_size', type=int, default=None, help='Batch size for batch scrubbing')

    return parser

if __name__ == '__main__':
    main()

