set  -e

use_case=$1
prefix=$2
#onto2graph
echo "onto2graph"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.onto2graph_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p onto2graph

#owl2vec
echo "owl2vec"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.owl2vec_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p owl2vec

#rdf
echo "rdf"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.rdf_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p rdf

# cat
echo "cat"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.cat_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p cat

#cat.s1
echo "cat.s1"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.cat.s1_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p cat

#cat.s2
echo "cat.s2"
python check_node_existence.py ../../use_cases/${use_case}/data/${prefix}.cat.s2_initial_terminal_no_leakage.edgelist ../../use_cases/${use_case}/data/classes.txt -p cat

