#!/bin/bash
echo 'Running Scrubbing Script for Retraining MNIST...'

# startID=$1
# endID=$2
# HessType=$3
# cuda_id=$4

startID=0
endID=10
HessType='Sekhari'
cuda_id=1

dataset='mnist'
model='Logistic2NN'
epochs=5

lr=0.001
batch_size=256
momentum=0.9
optimizer='sgd'

orig_trainset_size=50000
nRemovals=1000
scrubType='IP'
weight_decay=0.01
order='Hessian'
FOCIType='full'

approxType='FD'
n_perturbations=1000

val_gap_skip=0.05

hessian_device='cpu'

delta=0.01
epsilon=0.1

outfile='results/reatrain_scrub_mnist_2nn.csv'

CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
                    --epochs $epochs --optim $optimizer --learning_rate $lr --weight_decay $weight_decay --momentum $momentum \
                    --batch_size $batch_size

run () {

CUDA_VISIBLE_DEVICES=$cuda_id python retrain_scrub.py --dataset $dataset \
                --model $model \
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
                --val_gap_skip $val_gap_skip \
                --train_lr $lr \
                --train_wd $weight_decay \
                --train_momentum $momentum \
                --train_bs $batch_size \
                --train_optim $optimizer \

}

for runID in $(seq $startID 1 $endID)
    do
        for selectionType in 'FOCI'
            do
                run $runID $selectionType &
            done
        wait
    done

