set -e

use_case=$1
prefix=$2
#onto2graph
echo "onto2graph"
#python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.onto2graph.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p onto2graph

#owl2vec
echo "owl2vec"
#python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.owl2vec.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p owl2vec

#rdf
echo "rdf"
#python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.rdf.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p rdf

# cat
echo "cat"
python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.cat.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p cat

#cat.s1
echo "cat.s1"
python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.cat.s1.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p cat

#cat.s2
echo "cat.s2"
python add_initial_terminal.py -i ../../use_cases/${use_case}/data/${prefix}.cat.s2.edgelist -c ../../use_cases/${use_case}/data/classes.txt -p cat
