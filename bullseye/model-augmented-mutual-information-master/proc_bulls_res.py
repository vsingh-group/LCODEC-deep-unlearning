import argparse

import pandas as pd

import sys
sys.path.append('../../')
import utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--resfile', type=str, default='roc_results.csv')
args = parser.parse_args()

fname = args.resfile

df = pd.read_csv(fname)

df = df[df['featmaps']==True]

grouped = df.groupby(['model','conf','featmaps']).agg("mean")

print(grouped)

tprs = []
#tprs.append(df[df['model']=='codec'].groupby(['conf']).agg('mean')['tpr'])
tprs.append(df[df['model']=='codeccit'].groupby(['conf']).agg('mean')['tpr'])
tprs.append(df[df['model']=='bullscit'].groupby(['conf']).agg('mean')['tpr'])
fprs = []
#fprs.append(df[df['model']=='codec'].groupby(['conf']).agg('mean')['fpr'])
fprs.append(df[df['model']=='codeccit'].groupby(['conf']).agg('mean')['fpr'])
fprs.append(df[df['model']=='bullscit'].groupby(['conf']).agg('mean')['fpr'])

#ut.ROCPlot(tprs, fprs, outPath="", plotName="rocplot_bulls", fprtpr=True, legend=['codec','codeccit', 'bullscit'])
ut.ROCPlot(tprs, fprs, outPath="", plotName="rocplot_bulls", fprtpr=True, legend=['codeccit', 'bullscit'])
