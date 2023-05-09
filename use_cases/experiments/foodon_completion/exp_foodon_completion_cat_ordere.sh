python ../../run_model.py -case foodon_comp -kge ordere -g cat -r ../../foodon_completion/data -dim 128 -m 0.04 -wd 0 -bs 8192 -lr 0.0001 -tbs 16 -e 4000 -d cuda -rf result_foodon_cat_completion_ordere.csv -tf ../../foodon_completion/data/test.csv -vf ../../foodon_completion/data/valid.csv -tc

