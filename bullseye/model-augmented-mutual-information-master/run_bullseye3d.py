import numpy as np

import h5py
import pickle
import json
import torch

import os
import sys
sys.path.append("../pycit-master/")

import argparse

from pycit import *
from bullseye import bullseye_network, get_ci_dict
from mapping import ModelManager

# from graph_synth import createFOCIGraph

parser = argparse.ArgumentParser()
parser.add_argument('--foobar', action='store_true', help="True means use hypothesis testing for CODEC")
args = parser.parse_args()

if args.foobar:
	print("Do hypothesis testing for CODEC")
else:
	print("Use plain CODEC values")

# make data with illustrated DAG structure
num_samples = 5000
dim = 3
print('generating data...')
r_data, x_data, y_data = bullseye_network(num_samples, dim, eps=0.075)
r_data = standardize(r_data.astype(np.float16))
x_data = standardize(x_data.astype(np.float16))
y_data = standardize(y_data.astype(np.float16))

#n = 1000
#p = 9
#X = np.random.randn(n,p)

#print('Truth: 0>1, 1>2, 3>4, 3,4>5, 2,5>6, 6,7>8')

#X[:,1] = np.sin(X[:,0])
#X[:,2] = np.sqrt(np.sqrt(X[:,1]**2))

#X[:,4] = X[:,3] + 0.1*np.random.randn(n)
#X[:,5] = X[:,4]**3 + X[:,3]

#X[:,6] = X[:,5]*X[:,2]

#X[:,8] = np.cos(X[:,6]) + X[:,7]

#XY = (X-np.mean(X))/np.std(X)

#x_data = np.expand_dims(X[:,:8], axis=1)
#y_data = np.expand_dims(X[:,8], axis=1)

np.save("data/bullseye3dY.npy", y_data)
np.save("data/bullseye3dX.npy", x_data)

# # learn feature maps
config_path = "config/bullseye3d.json"
manager = ModelManager(config_path, make_plots=True)
print('training...')
manager.train()

import pdb; pdb.set_trace()

# # load learned feature mappings
# manager.load_model(checkpoint_file="model_best.pth")
# mu,lv,z = manager.process_numpy(x_data)
# z_data = standardize(z)

# print("X(Original data): ", x_data.shape)
# print("Y(Target): ", y_data.shape)
# print("Z(Mapped data): ", z_data.shape)

# CIT settings
K_KNN = 5
K_PERM = 5
SUBSAMPLE_SIZE = None
N_TRIALS = 250
N_JOBS = 1

codec_cit_funcs = {
    'it_args': {
    	'statistic': 'codec2',
        'statistic_args': {
            'k': K_KNN
        },
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
        'statistic_args': {
            'k': K_KNN
        },
        'test_args': {
            'statistic': 'codec3',
            'k_perm': K_PERM,
            'subsample_size': SUBSAMPLE_SIZE,
            'n_trials': N_TRIALS,
            'n_jobs': N_JOBS
        }
    }
}

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

# Use high confidence value, if using hypothesis testing for codec
# print("find Markov Blanket with original data and CODEC")
# mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
# selected = mb.find_markov_blanket(confidence=0.95, verbose=True, codec_hyp=args.foobar)

# Use low confidence value, if using direct codec values for selection process
#print("find Markov Blanket with original data and CODEC")
#mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
#selected = mb.find_markov_blanket(confidence=0.001, verbose=True, codec_hyp=args.foobar)

X = np.squeeze(x_data)
Y = np.expand_dims(np.squeeze(y_data), axis=1)
XY = np.hstack([X, Y])
print(XY.shape)

outputGraph = createFOCIGraph(XY)
print(outputGraph)
print(outputGraph>0)

# print("find Markov Blanket with original data and CODEC")
# mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
# selected = mb.find_markov_blanket(confidence=0.001, verbose=True, codec_hyp=args.foobar)

# print("find Markov Blanket with original data and CODEC")
# mb = MarkovBlanket(x_data, y_data, codec_cit_funcs)
# selected = mb.find_markov_blanket(confidence=0.001, verbose=True, codec_hyp=args.foobar)

#print("find Markov Blanket with original data and CMI")
#mb = MarkovBlanket(x_data, y_data, cmi_cit_funcs)
#selected = mb.find_markov_blanket(confidence=0.95, verbose=True, codec_hyp=args.foobar)

# print("find Makov Blanket using mapped features and CMI")
# mb = MarkovBlanket(z_data, y_data, cmi_cit_funcs)
# selected = mb.find_markov_blanket(confidence=0.95, verbose=True, codec_hyp=args.foobar)
