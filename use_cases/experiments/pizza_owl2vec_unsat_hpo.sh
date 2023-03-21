set -e

for dim in 32 64 128
do
    for margin in 0.0 0.1 0.2
    do
	for wd in 0.0 0.0001 0.0005
	do
	    for bs in 128 256 512
	    do
		for lr in 0.001 0.0001 0.00001
		do
		    python run_model.py -case pizza -kge transe -g owl2vec -r pizza/data -dim $dim -m $margin -wd $wd -bs $bs -tbs 8 -e 4000 -d cuda -rf result_pizza_owl2vec_unsattranse_hpo_2.csv -tf pizza/data/pizza_unsat_classes.csv -tu -ot
		done
	    done
	done
    done
done

