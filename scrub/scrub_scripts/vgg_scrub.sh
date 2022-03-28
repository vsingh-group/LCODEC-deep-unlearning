#!/bin/bash
echo 'Running Scrubbing Script for VGGFace with Pretrained model...'

export CUDA_VISIBLE_DEVICES=0,1
echo $CUDA_VISIBLE_DEVICES

dataset='vggface'
model='VGG_16'
epochs=4 # just to get the gradients

optim=sgd
lr=0.0001
batch_size=16
weight_decay=0.01

echo $MODEL_FILE

id_to_scrub='Aamir_Khan'
nRemovals=3 # subset from id
scrub_batch_size=1 #batch to scrube at once
orig_trainset_size=982803 # original taken from 2015 paper (including valid?)

scrubType='IP'
order='Hessian'
HessType='Sekhari'
selectionType='FOCI'
FOCIType='cheap'
approxType='FD'
n_perturbations=5
delta=0.01
epsilon=0.0001
hessian_device='cpu'
sec_device='cuda:1'

run=1

outfile="results/test_vggface_scrub_${id_to_scrub}_eps_${epsilon}.csv"

MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
echo $MODEL_FILE

# This training is to estimate the Hessian of the model over the training set
# python train.py --dataset $dataset --model $model --epochs $epochs \
#         --batch_size 16 --weight_decay $weight_decay --learning_rate $lr \
#          --optim $optim --batch_size $batch_size

python vgg_scrub.py --dataset $dataset \
                --model $model \
                --MODEL_FILE $MODEL_FILE \
                --orig_trainset_size $orig_trainset_size \
                --train_epochs $epochs \
                --batch_size $batch_size \
                --run $run \
                --order $order \
                --selectionType $selectionType \
                --FOCIType $FOCIType \
                --HessType $HessType \
                --approxType $approxType \
                --scrubType $scrubType \
                --l2_reg $weight_decay \
                --n_perturbations $n_perturbations \
                --n_removals $nRemovals \
                --delta $delta \
                --epsilon $epsilon \
                --hessian_device $hessian_device \
                --sec_device $sec_device \
                --id_to_scrub $id_to_scrub \
                --outfile $outfile \
                #--scrub_batch_size $scrub_batch_size \
