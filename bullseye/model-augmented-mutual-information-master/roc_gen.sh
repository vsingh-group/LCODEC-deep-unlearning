#!/bin/bash
echo 'Generating runs for ROC plot for Bullseye Experiments...'

nruns=10


bullsrun () {
echo Starting new process with $1 $2 $3 $4
python roc_gen.py --run $1 --data bullseye3d --model $2 --conf $3 --n_samples 5000 --n_trials 100 --outfile roc_results_test_feat.csv $4
}

for run in $(seq 1 1 $nruns)
    do
        #for conf in 0.01 0.05 0.1 0.5 0.9 0.95
        for conf in 0.05 0.1 0.5 0.9
            do
                for model in codec bullscit codeccit
                #for model in codec 
                    do
                        bullsrun $run $model $conf #&
                        bullsrun $run $model $conf --feat_maps #&
                    done
                #wait
                echo Done with $model $conf
            done
    done

