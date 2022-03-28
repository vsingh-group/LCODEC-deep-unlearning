#!/bin/bash
echo 'Running CelebA models with GRL for Selected Spurious Feautres'
do_per_core () {
echo Starting new process with attribute $1 $2 $3
python tester.py --run $1 --selectionType $2 --reg_strength $3 --trainset_size=30000 --epochs 10 --noise 0.01
}

# single run for no reg
do_per_core 1 None 0.0
wait
echo Done with No Reg Run


# random runs for different reg strengths
for i in 0.001 0.01 0.1 1.0 
	do
		do_per_core 1 Random $i &
	done
wait
echo Done with Random Reg Runs

# foci runs for different reg strengths
for i in 0.001 0.01 0.1 1.0 
	do
		do_per_core 1 FOCI $i &
	done
wait
echo Done with FOCI Reg Runs

# full runs for different reg strengths
for i in 0.001 0.01 0.1 1.0 
	do
		do_per_core 1 All $i &
	done
wait
echo Done with Full Reg Runs
