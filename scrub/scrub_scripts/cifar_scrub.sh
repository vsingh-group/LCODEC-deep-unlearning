#!/bin/bash
echo 'Running CIFAR MultiModel Scrubbing Script...'

dataset='cifar10'
models="densenet mobilenet resnet18 resnet50 vgg11_bn CIFAR10Logistic2NN"
ModelList=($models)
optims="sgd sgd sgd sgd sgd sgd"
OptimList=($optims)
train_optim='sgd'

augmentChoice=(0 0 0 0 0 0)

epochs=200

lrList=(0.1 0.1 0.1 0.1 0.01 0.01)

batch_size=32

orig_trainset_size=50000
scrubType='IP'
order='Hessian'
selectionType='FOCI'
FOCIType='full'
approxType='FD'

weight_decay=0.01
lreg=0.01

delta=0.01
epsilon=0.1

hessian_device='cpu'

nRemovals=1000
n_perturbations=1000
nRuns=1

outfile='results/cifar_scrub_results.csv'
train () {

	model=${ModelList[$1]}
	optim=${OptimList[$1]}
        lr=${lrList[$1]}
	MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"

	echo $MODEL_FILE

	CUDA_VISIBLE_DEVICES=$1 python train.py --dataset $dataset --model $model --epochs $epochs --weight_decay $weight_decay --learning_rate $lr --optim $optim --batch_size $batch_size
}

scrub () {
        augment=${augmentChoice[$2]}
        lr=${lrList[$2]}
		model=${ModelList[$2]}
		MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
		echo $MODEL_FILE

		CUDA_VISIBLE_DEVICES=$2 python multi_scrub.py --dataset $dataset \
				--model $model \
				--orig_trainset_size $orig_trainset_size \
				--train_epochs $epochs \
				--batch_size $batch_size \
				--run $1 \
				--order $order \
				--selectionType $selectionType \
	            --FOCIType $FOCIType \
				--HessType $3 \
				--approxType $approxType \
				--scrubType $scrubType \
				--l2_reg $lreg \
				--n_perturbations $n_perturbations \
				--n_removals $nRemovals \
				--delta $delta \
				--epsilon $epsilon \
	            --hessian_device $hessian_device \
				--outfile $outfile \
	            --train_lr $lr \
	            --train_wd $weight_decay \
	            --train_bs $batch_size \
	            --train_optim $train_optim \
	            --data_augment $augment
}

for model_idx in $(seq 0 1 5)
   do
   train $model_idx &
   done

wait

for runID in $(seq 1 1 $nRuns)
    do
    for HessType in 'Sekhari'
        do
        for model_idx in $(seq 0 1 5)
            do
            scrub $runID $model_idx $HessType &
            done
        done
        wait
    done

