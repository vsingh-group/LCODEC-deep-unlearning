#!/bin/bash
echo 'Running CelebA models with GRL for Selected Spurious Feautres'
do_per_core () {
echo Starting new process with attribute $1 $2 $3
python spureg_train.py --run $1 --selectionType $2 --reg_strength $3 --trainset_size=30000 --epochs 30 --noise 0.01
}

# single run for no reg
do_per_core 1 None 0.0

# random runs
for myrunid in $(seq 1 1 6)
	do
		do_per_core $myrunid Random 0.1 &
	done
wait
echo Done with Half Random Runs

for myrunid in $(seq 7 1 14)
	do
		do_per_core $myrunid Random 0.1 &
	done
wait
echo Done with Half Random Runs

for myrunid in $(seq 15 1 20)
	do
		do_per_core $myrunid Random 0.1 &
	done
wait
echo Done with all Random Runs

# foci runs for different reg strengths
for i in 0.0001 0.001 0.01 0.1 1.0 10.0 100.0
	do
		do_per_core 1 FOCI $i &
	done
wait
echo Done with FOCI Reg Runs

# full runs for different reg strengths
for i in 0.0001 0.001 0.01 0.1 1.0 10.0 100.0
	do
		do_per_core 1 All $i &
	done
wait
echo Done with Full Reg Runs
