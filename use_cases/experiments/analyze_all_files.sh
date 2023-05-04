#!/bin/bash

set -e
# prefix of the files to process
prefix='result'

# path to the directory containing the files
dirpath=$1
final_metric=$2

# path to the Python script to execute over each file
scriptpath="analyze_results_all.py"

# loop over the files in the directory with the prefix
for filepath in ${dirpath}/${prefix}*
do
    # execute the Python script with the file path
    echo $filepath
    python ${scriptpath} ${filepath} ${final_metric}
done
