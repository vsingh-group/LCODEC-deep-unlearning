import csv
import numpy as np 
import time
import os
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from viz import get_foci_matrix

parser = argparse.ArgumentParser(description='Other Correleation measures')
parser.add_argument('--noise', default=0.01, type=float, help='Noise')
parser.add_argument('--n_samples', default=20000, type=int, help='Number of samples')
parser.add_argument('--quality', default=200, type=int, help='Picture dpi')
# parser.add_argument('--cor_type', default='pearsonr', type=str, help='Correleation measures: Pearson, Spearman, Kendalltau')
args = parser.parse_args()

# print("Correleation measure: ", args.cor_type)
print("Noise level: ", args.noise)

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
print("Data shape: ", attr_data.shape)   
attr_data = attr_data.astype(np.float)

# Add noise
no_noise_attr_data = np.copy(attr_data)
attr_data = low_amplitude_noise(attr_data, eps=args.noise)
n_attrs = attr_data.shape[1]

def get_corr_matrix(n_attrs, cor_type, attr_data):
    corr_matrix = np.zeros((n_attrs, n_attrs))

    for x_ind in range(n_attrs):
        for y_ind  in range(n_attrs):
            if cor_type == 'pearsonr':
                corr_val, p_val = scipy.stats.pearsonr(attr_data[:,x_ind], attr_data[:,y_ind])
                corr_matrix[x_ind][y_ind] = np.abs(corr_val)
            elif cor_type == 'spearmanr':
                corr_val, p_val = scipy.stats.spearmanr(attr_data[:,x_ind], attr_data[:,y_ind])
                corr_matrix[x_ind][y_ind] = np.abs(corr_val)
            elif cor_type == 'kendalltau':
                corr_val, p_val = scipy.stats.kendalltau(attr_data[:,x_ind], attr_data[:,y_ind])
                corr_matrix[x_ind][y_ind] = np.abs(corr_val)
            else:
                print("Correleation measure not defined")
                break

    return corr_matrix
    

# fig, ax = plt.subplots(figsize=(15,15))
# svm = sns.heatmap(corr_matrix, cmap="YlGnBu", xticklabels=header, yticklabels=header, ax=ax, linewidths=0.2, linecolor='black')

# figure = svm.get_figure()  

# dirname = "other_corr_mea_noise_"+str(args.noise)
# if not os.path.isdir(dirname):
#     os.makedirs(dirname)

# fig_name = dirname + "/" + "attr_"+args.cor_type+"_noise_"+str(args.noise)+".png"
# figure.savefig(fig_name, dpi=args.quality, bbox_inches="tight")


def grid_plotter(n_attrs, attr_data, no_noise_attr_data, header, fname, args):
    ncolumns = 3
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=[30,30])

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if i==0:
            svm = sns.heatmap(get_corr_matrix(n_attrs, 'pearsonr', no_noise_attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=header, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        elif i==1:
            svm = sns.heatmap(get_corr_matrix(n_attrs, 'spearmanr', no_noise_attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        elif i==2:
            foci_matrix, fnames = get_foci_matrix(0.0)
            svm = sns.heatmap(foci_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        elif i==3:
            svm = sns.heatmap(get_corr_matrix(n_attrs, 'pearsonr', attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=header, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        elif i==4:
            svm = sns.heatmap(get_corr_matrix(n_attrs, 'spearmanr', attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        elif i==5:
            foci_matrix, fnames = get_foci_matrix(args.noise)
            svm = sns.heatmap(foci_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axi, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": .4})
        else:
            print("Invalid")
            break

    plt.tight_layout()
    plt.savefig(fname, dpi=args.quality, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()
    return

fname = "attr_cor_noise_0_and_"+str(args.noise)+".png"
grid_plotter(n_attrs, attr_data, no_noise_attr_data, header, fname, args)
print("All Plotting Done!!")


fig, ax = plt.subplots(figsize=(30,30))
svm = sns.heatmap(get_corr_matrix(n_attrs, 'pearsonr', no_noise_attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=header, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("Pearson_noise_0.png", dpi=args.quality, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(30,30))
svm = sns.heatmap(get_corr_matrix(n_attrs, 'spearmanr', no_noise_attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("Spearman_noise_0.png", dpi=args.quality, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(30,30))
foci_matrix, fnames = get_foci_matrix(0.0)
svm = sns.heatmap(foci_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("CODEC_noise_0.png", dpi=args.quality, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(30,30))
svm = sns.heatmap(get_corr_matrix(n_attrs, 'pearsonr', attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=header, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("Pearson_noise_"+str(args.noise)+".png", dpi=args.quality, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(30,30))
svm = sns.heatmap(get_corr_matrix(n_attrs, 'spearmanr', attr_data), cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("Spearman_noise_"+str(args.noise)+".png", dpi=args.quality, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(30,30))
foci_matrix, fnames = get_foci_matrix(args.noise)
svm = sns.heatmap(foci_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=ax, linewidths=0.2, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
sin_fig = svm.get_figure()  
sin_fig.savefig("CODEC_noise_"+str(args.noise)+".png", dpi=args.quality, bbox_inches="tight")