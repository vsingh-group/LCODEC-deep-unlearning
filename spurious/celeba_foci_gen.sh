#!/bin/bash
echo 'Doing FOCI on celebA attributes'
do_per_core () {
echo Hello starting new core with attribute $1 and noise $2
python celeb_attr_foci.py --attr $1 --noise $2 --n_samples 20000
}

echo Starting Noise level 0.00001
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.00001 &
	done

wait
echo Done with Noise level 0.00001

echo Starting Noise level 0.0001
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.0001 &
	done

wait
echo Done with Noise level 0.0001

echo Starting Noise level 0.001
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.001 &
	done

wait
echo Done with Noise level 0.001

echo Starting Noise level 0.01
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.01 &
	done

wait
echo Done with Noise level 0.01



echo Starting Noise level 0.1
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.1 &
	done

wait
echo Done with Noise level 0.1

echo Starting Noise level 0.0
for i in {1..40}
	do
		attribute=$(($i-1))
		do_per_core $attribute 0.0 &
	done

wait
echo Done with Noise level 0.0