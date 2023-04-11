set -e

use_case=$1
prefix=$2
suffix=$3

#onto2graph
echo "onto2graph"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.onto2graph${suffix}edgelist ../../use_cases/${use_case}/data/test.csv onto2graph

#owl2vec
echo "owl2vec"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.owl2vec${suffix}edgelist ../../use_cases/${use_case}/data/test.csv owl2vec

#rdf
echo "rdf"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.rdf${suffix}edgelist ../../use_cases/${use_case}/data/test.csv rdf

#cat
echo "cat"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.cat${suffix}edgelist ../../use_cases/${use_case}/data/test.csv cat

#cat.s1
echo "cat.s1"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.cat.s1${suffix}edgelist ../../use_cases/${use_case}/data/test.csv cat

#cat.s2
echo "cat.s2"
python remove_leakage.py ../../use_cases/${use_case}/data/${prefix}.cat.s2${suffix}edgelist ../../use_cases/${use_case}/data/test.csv cat
