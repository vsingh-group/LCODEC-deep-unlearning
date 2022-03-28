#!/bin/bash
echo 'Running Scrubbing Script for some Ablations...'

startID=$1
endID=$2
HessType=$3
cuda_id=$4

dataset='mnist'
model='Logistic2NN'
epochs=50

lr=0.001
batch_size=256

echo $MODEL_FILE

orig_trainset_size=50000
nRemovals=1000
scrubType='IP'
weight_decay=0.01
order='Hessian'
FOCIType='full'
cheap_foci_thresh=0.05

approxType='FD'
n_perturbations=1000

val_gap_skip=0.05

hessian_device='cpu'

delta=0.01
epsilon=0.1

outfile='results/mnist_2nn.csv'

MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
echo $MODEL_FILE

CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
                    --epochs $epochs --weight_decay $weight_decay \
                    --batch_size $batch_size

run () {

CUDA_VISIBLE_DEVICES=$cuda_id python multi_scrub.py --dataset $dataset \
                --model $model \
                --MODEL_FILE $MODEL_FILE \
                --orig_trainset_size $orig_trainset_size \
                --batch_size $batch_size \
                --train_epochs $epochs \
                --run $1 \
                --order $order \
                --selectionType $2 \
                --HessType $HessType \
                --approxType $approxType \
                --scrubType $scrubType \
                --l2_reg $weight_decay \
                --n_perturbations $n_perturbations \
                --n_removals $nRemovals \
                --delta $delta \
                --epsilon $epsilon \
                --outfile $outfile \
                --hessian_device $hessian_device \
                --val_gap_skip $val_gap_skip
}

for runID in $(seq $startID 1 $endID)
    do
        for selectionType in 'Full' 'Random' 'FOCI'
            do
                run $runID $selectionType &
            done
        wait
    done

