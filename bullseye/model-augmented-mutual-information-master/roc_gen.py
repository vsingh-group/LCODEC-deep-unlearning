import numpy as np
import pandas as pd
import time

import h5py
import pickle
import json
import torch

import os
import sys
sys.path.append("../pycit-master/")
sys.path.append("../../")
from codec import codec2, codec3, foci

import argparse

from pycit import *
from bullseye import bullseye_network, get_ci_dict
from mapping import ModelManager

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['bullscit','codeccit','codec','sdcit', 'ccit','foci'], default='codec')
parser.add_argument('--data', type=str, choices=['bullseye3d','customDAG'], default='bullseye3d')
parser.add_argument('--feat_maps', default=False, action='store_true', help="True means use hypothesis testing for CODEC")
parser.add_argument('--ckptfile', default='saved/3d_bullseye_regularized/0525_121114/model_best.pth', type=str, help="checkpoint for loading featuremaps")
parser.add_argument('--conf', default=0.1, type=float, help="confidence threshold for CI tests")
parser.add_argument('--run', default=0, type=int, help="run ID")
parser.add_argument('--n_trials', default=100, type=int, help="number of permutations for CI null distribution")
parser.add_argument('--n_samples', default=5000, type=int, help="number of samples from graphs")
parser.add_argument('--outfile', default='results.csv', type=str, help="csv to append results to")
parser.add_argument('--verbose', default=False, type=bool, help="verbosity on/off")
args = parser.parse_args()

tmp = {}
tmp['n_trials'] = [args.n_trials]
tmp['model'] = [args.model]
tmp['data'] = [args.data]
tmp['nrun'] = [args.run]
tmp['conf'] = [args.conf]
tmp['n_samples'] = [args.n_samples]
tmp['featmaps'] = [args.feat_maps]

num_samples = args.n_samples
# CIT settings
K_KNN = 5
K_PERM = 5
SUBSAMPLE_SIZE = None
N_JOBS = 1
N_TRIALS = args.n_trials

if args.data == 'bullseye3d':
    # BULLSEYE
    # make data with illustrated DAG structure
    dim = 3
    _, x_data, y_data = bullseye_network(num_samples, dim, eps=0.075)
    x_data = standardize(x_data.astype(np.float16))
    y_data = standardize(y_data.astype(np.float16))
    np.save("data/bullseye3dY.npy", y_data)
    np.save("data/bullseye3dX.npy", x_data)

    trueset = set([2,3,4,5])
    falseset = set([0,1])

elif args.data == 'customDAG':
    # SampleDAG
    x_data, y_data = genSamplesfromDAG(num_samples)
else:
    error('unknown data')

if args.feat_maps:
    # learn feature maps
    config_path = "config/bullseye3d.json"
    manager = ModelManager(config_path, make_plots=True, is_resume=False)
    #manager.train()

    # load learned feature mappings
    mu,lv,z_data = manager.process_numpy(x_data, checkpoint_file=args.ckptfile, fullpath=True)
    x_data = standardize(z_data)

    if args.verbose:
        print("X(Original data): ", x_data.shape)
        print("Y(Target): ", y_data.shape)
        print("Z(Mapped data): ", z_data.shape)

    x_data = z_data


begintime = time.time()

if args.model == 'bullscit':
    cmi_cit_funcs = {
        'it_args': {
            'statistic': 'mixed_mi',
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_mi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        },
        'cit_args': {
            'statistic': 'mixed_cmi',
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_cmi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        }
    }
    
    if args.verbose:
        print("find Markov Blanket with mapped data and CODEC")
    mb = MarkovBlanket(x_data, y_data, cmi_cit_funcs)
    selected = mb.find_markov_blanket(confidence=args.conf, verbose=False, codec_hyp=True)

elif args.model == 'codeccit':
    codec_cit_funcs = {
        'it_args': {
            'statistic': 'codec2',
            'test_args': {
                'statistic': 'codec2',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS } },
        'cit_args': {
            'statistic': 'codec3',
            'test_args': {
                'statistic': 'codec3',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        }
    }

    if args.verbose:
        print("find Markov Blanket with mapped data and CODEC")
    mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
    selected = mb.find_markov_blanket(confidence=args.conf, verbose=False, codec_hyp=True)

elif args.model == 'codec':
    codec_cit_funcs = {
        'it_args': {
            'statistic': 'codec2',
            'test_args': {
                'statistic': 'codec2',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        },
        'cit_args': {
            'statistic': 'codec3',
            'test_args': {
                'statistic': 'codec3',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        }
    }

    if args.verbose:
        print("find Markov Blanket with mapped data and CODEC")
    mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
    selected = mb.find_markov_blanket(confidence=args.conf, verbose=False, codec_hyp=False)

elif args.model == 'foci':
    if x_data.shape[1] == 3:
        x_data = np.reshape(x_data, [15000, 6])
        y_data = np.repeat(y_data, 3)
    elif x_data.shape[1] == 1:
        x_data = np.squeeze(x_data)

    ordering, values = foci(x_data, y_data, verbose=True)
    selected = ordering
    print('foci ordering: ', ordering)


elif args.model == 'ccit':
    error('unknown model')

elif args.model == 'sdcit':
    error('unknown model')

else:
    error('unknown model')

#tmp['selected'] = [selected]
tmp['tpr'] = [len(set(selected) & trueset)/len(trueset)]
tmp['fpr'] = [len(set(selected) & falseset)/len(falseset)]

tocd = time.time()-begintime
tmp['runtime'] = [tocd]

if args.verbose:
    print(f'Took {tocd} seconds.')
    print(tmp)

df = pd.DataFrame(tmp)
#fname = f'res/results_{args.data}_{args.model}_{args.conf}_{args.run}.csv'
if os.path.isfile(args.outfile):
    df.to_csv(args.outfile, mode='a', header=False, index=False)
else:
    df.to_csv(args.outfile, mode='a', header=True, index=False)
