#!/bin/bash
eval "echo -e 'number_of_objects, valid_error, exact_error, RMSE, CPU_time_to_train\n' >> mean_perf_ESN.txt"
for nb_obj in {3..50}
do
	eval "python ../CSL_ESN.py ${nb_obj} 1 >> mean_perf_ESN.txt"
	echo "${nb_obj} objects: done."
done
