import csv
import numpy as np 
import time
import os
import argparse

import sys
sys.path.append('..')
from codec import foci

parser = argparse.ArgumentParser(description='FOCI CELEBA Attribute Markov Blanket')
parser.add_argument('--attr', default=0, type=int, help='Attribute ID')
parser.add_argument('--noise', default=0.0, type=float, help='CODEC Noise')
parser.add_argument('--n_samples', default=20000, type=int, help='Number of samples')
args = parser.parse_args()

def low_amplitude_noise(data, eps=1e-10):
    """
        Add low amplitude noise in order to break ties
    """
    return data + np.random.normal(0.0, eps, size=data.shape)

with open("./data/celeba/list_attr_celeba.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    all_data = []
    for row in csv_reader:
        row = row[0].split(",")
        all_data.append(row)
        line_count +=1
    csv_file.close()
    header = all_data[0][1:]
    data = np.asarray(all_data[1:])

attr_data = data[:,1:]
np.random.shuffle(attr_data)
attr_data = attr_data[:args.n_samples,:]
# print(attr_data.shape, flush=True)   
attr_data = attr_data.astype(np.float)

# Add noise
attr_data = low_amplitude_noise(attr_data, eps=args.noise)

index = args.attr
# print("Attribute Index: ", index, flush=True)
header = np.asarray(header)

dirname = "noise_level_"+str(args.noise)
if not os.path.isdir(dirname):
    os.makedirs(dirname)
fname = dirname + "/" + str(index)+"_"+header[[index]][0]+".npy"

header = np.delete(header, index)

# print("Starting FOCI", flush=True)
foci_Y = attr_data[:,index]
foci_X = np.delete(attr_data, index, 1)
# print("FOCI_X Shape", foci_X.shape)
# print("FOCI_Y Shape", foci_Y.shape)
tic = time.time()
ordering, scores = foci(foci_X, foci_Y, earlyStop=True, verbose=False)
time_taken = time.time() - tic
# print(ordering)
# print(header[ordering])
# print(scores)
# print('Took: {} seconds.'.format(time.time()-tic))

# This increment is required because of the reduced dimension 
# and hence indexing of the features that are after the current target
for k in range(len(ordering)):
    stored_val = ordering[k]
    if stored_val>= index:
        ordering[k] = stored_val+1

storage = []
storage.append(ordering)
storage.append(header[ordering])
storage.append(scores)
storage.append(time_taken)
storage = np.asarray(storage)
print("Storing numpy ", fname)
np.save(fname, storage)