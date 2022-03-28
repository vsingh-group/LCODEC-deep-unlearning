import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate

def fill_column(key, args, target_dir):
    current_col = []
    row_ids = []
    
    noreg_run = 1
    noregname = target_dir + "None_0.0_" + str(noreg_run) +  ".csv"
    noregdict = pd.read_csv(noregname)
    current_col.append(noregdict[key].iloc[args.val_epoch])
    row_ids.append("None")

    # for run in range(20):
    #     randname = target_dir + "Random_" + "1.0" + "_" + str(run+1) + ".csv"
    #     randdict = pd.read_csv(randname)
    #     current_col.append(randdict[key].iloc[args.val_epoch])
    #     row_ids.append("Random_"+str(run+1))

    regstrs = [0.01, 0.1, 0.2, 0.5, 1.0]
    # regstrs = [0.001, 0.01, 0.1, 1.0]
    random_run = 1
    for regstr in regstrs:
        randname = target_dir + "Random_" + str(regstr) + "_" + str(random_run) + ".csv"
        randdict = pd.read_csv(randname)
        current_col.append(randdict[key].iloc[args.val_epoch])
        row_ids.append("Random_"+str(regstr))

    foci_run = 1
    for regstr in regstrs:
        fociname = target_dir + "FOCI_" + str(regstr) + "_" + str(foci_run) + ".csv"
        focidict = pd.read_csv(fociname)
        current_col.append(focidict[key].iloc[args.val_epoch])
        row_ids.append("FOCI_"+str(regstr))

    all_run = 1
    for regstr in regstrs:
        allname = target_dir + "All_" + str(regstr) + "_" +  str(all_run) + ".csv"
        alldict = pd.read_csv(allname)
        current_col.append(alldict[key].iloc[args.val_epoch])
        row_ids.append("All_"+str(regstr))

    return current_col, row_ids

arg_parser = argparse.ArgumentParser(description='CelebA FOCI Runs')
arg_parser.add_argument('--target', type=str, default="Young")
arg_parser.add_argument('--key', type=str, default='val_full_acc')
arg_parser.add_argument('--noise', type=float, default=0.01, help="Noise level for generaing the FOCI selections")
arg_parser.add_argument('--trainset_size', type=int, default=30000, help="Total trainset set size used")
arg_parser.add_argument('--epochs', type=int, default=30, help="Total no.of epochs models were trained for")
arg_parser.add_argument('--val_epoch', type=int, default=-1, help="Epoch for which we want to report values")
args = arg_parser.parse_args()

target_dir = "Target_"+args.target +"_noise_"+str(args.noise)+"_train_"+str(args.trainset_size)+"_epoch_"+str(args.epochs) +"/"
print("Folder: ", target_dir)

attr_data = pd.read_csv("./data/celeba/list_attr_celeba.csv")

# removing image column from indexing
# this value should be 0 to 39
target_index = attr_data.columns.get_loc(args.target)-1
print("Target: ", args.target)
print("Target Index: ", target_index)

fname = "noise_level_"+str(args.noise)+"/"+str(target_index)+"_"+args.target+".npy"
foci_numpy = np.load(fname, allow_pickle=True)
foci_sel_attr_indexes = []
for attrs in foci_numpy[1]:
    temp_ind = attr_data.columns.get_loc(attrs)-1
    foci_sel_attr_indexes.append(temp_ind)

print("FOCI selection: ", foci_numpy[1])
print("FOCI selected scores: ", foci_numpy[2])
print("FOCI selected indices: ", foci_sel_attr_indexes)

assert foci_numpy[0] == foci_sel_attr_indexes

val_types = ['train', 'val']
val_conds = ['with', 'without']

for t in val_types:
    all_values = []
    all_headers = []

    print("\n\n"+"\t"*10+"*"*10+t+" values"+"*"*10)

    full_key = t+"_full_acc"
    temp_c, row_ids = fill_column(full_key, args, target_dir)
    all_values.append(temp_c)
    all_headers.append("full_acc")
    
    for index in foci_sel_attr_indexes:
        for c in val_conds:
        
            key = t+"_"+c+"_"+str(index)
            temp_c, _ = fill_column(key, args, target_dir)
            all_values.append(temp_c)
            if c == 'with':             
                all_headers.append("w_"+str(index))
            else:
                all_headers.append("wo_"+str(index))

    all_values = np.asarray(all_values).transpose()
    print(tabulate(all_values, headers=all_headers, showindex=row_ids, floatfmt=".5f"))         