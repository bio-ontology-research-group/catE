import sys
sys.path.append("../")
import pandas as pd
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm
from utils import bot_name
import os
import matplotlib.pyplot as plt
import numpy as np

root = "../../use_cases"

use_case = sys.argv[1]
graph_type = sys.argv[2]

graph_prefix = graph_type
if graph_type == "cat1":
    graph_prefix = "cat.s1"
if graph_type == "cat1c":
    graph_type = "cat1"
    graph_prefix = "cat.s1_cleaned"
elif graph_type == "cat2":
    graph_prefix = "cat.s2"
elif graph_type == "cat2c":
    graph_type = "cat2"
    graph_prefix = "cat.s2_cleaned"

root = os.path.join(root, use_case, "data")

edges_file = os.path.join(root, f"{use_case}.{graph_prefix}.edgelist")
assert os.path.exists(edges_file), "File not found: {}".format(edges_file)

test_set_file = os.path.join(root, "test.csv")
assert os.path.exists(test_set_file), "File not found: {}".format(test_set_file)

classes_file = os.path.join(root, "classes.txt")
assert os.path.exists(classes_file), "File not found: {}".format(classes_file)

 
bot = bot_name[graph_type]



# Create graph
edges = pd.read_csv(edges_file, header=None, sep='\t')
edges.columns = ["head", "rel", "tail"]
edges = edges.drop_duplicates()

G = nx.DiGraph()
heads = edges["head"].tolist()
tails = edges["tail"].tolist()
G.add_edges_from(zip(heads, tails))


# Unsatisfiable classes
test_set = pd.read_csv(test_set_file, header=None, sep=',')
test_set.columns = ["head", "tail"]
test_set["tail"] = test_set["tail"].apply(lambda x: bot if x == "owl:Nothing" else x)

unsat_classes = test_set["head"].unique().tolist()

# All classes
all_classes = pd.read_csv(classes_file, header=None, sep='\t')
all_classes.columns = ["class"]
all_classes = all_classes["class"].tolist()


with_path = 0
without_path = 0
node_not_found = 0
avg_path_length = 0
not_found_nodes = set()


def find_path(nodes):
    with_path, without_path, node_not_found, avg_path_length, not_found_nodes = 0, 0, 0, 0, set()
    first_node, second_node = nodes
    path = []
    try:
        path = nx.shortest_path(G, source=first_node, target=second_node)
        
        if len(path) > 1:
            with_path += 1
            avg_path_length += len(path)
        else:
            without_path += 1
    except nx.exception.NetworkXNoPath:
        without_path += 1
    except nx.exception.NodeNotFound:
        node_not_found += 1
        if first_node not in G.nodes:
            not_found_nodes.add(first_node)
        if second_node not in G.nodes:
            not_found_nodes.add(second_node)

    return with_path, without_path, node_not_found, avg_path_length, not_found_nodes, (first_node, len(path)), path

path_lengths = dict()

sat_paths = []
unsat_paths = []
max_path_length = 0
with mp.Pool(processes=16) as pool:
    max_ = len(all_classes)
    nodes_pairs = [(node, bot) for node in all_classes]
    with tqdm(total=max_) as pbar:
        for r in pool.imap_unordered(find_path, nodes_pairs):
            with_path += r[0]
            without_path += r[1]
            node_not_found += r[2]
            avg_path_length += r[3]
            not_found_nodes.update(r[4])
            node, path_length = r[5]
            if path_length > max_path_length:
                max_path_length = path_length
            path = r[6]
            path_lengths[node] = path_length
            if node in unsat_classes:
                unsat_paths.append(path)
            else:
                sat_paths.append(path)
            pbar.update()

avg_path_length = avg_path_length / (with_path)

print("With path: ", with_path)
print("Average path length: ", avg_path_length)

print("Without path: ", without_path)
print("Node not found: ", node_not_found, len(not_found_nodes))

print("Sat Paths:")
for path in sat_paths:
    print(path)
print("Unsat Paths:")
for path in unsat_paths:
    print(path)

# plot path length distribution using matplotlib
unsat_class_dists = [path_lengths[unsat_class] for unsat_class in unsat_classes]
unsat_class_dists = [max_path_length + 1 if dist == 0 else dist for dist in unsat_class_dists]

sat_class_dists = [path_lengths[sat_class] for sat_class in all_classes if sat_class not in unsat_classes]
sat_class_dists = [max_path_length + 1 if dist == 0 else dist for dist in sat_class_dists]

plt.hist([sat_class_dists, unsat_class_dists], color=['blue', 'red'], label=['Satisfiable', 'Unsatisfiable'])
plt.title("Path length distribution")
plt.xlabel("Path length")
plt.ylabel("Frequency")
plt.show()




