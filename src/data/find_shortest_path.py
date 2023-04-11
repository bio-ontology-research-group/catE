import sys
import pandas as pd
import networkx as nx

input_file = sys.argv[1]
first_node = sys.argv[2]
second_node = sys.argv[3]

if first_node.startswith('GO'):
    first_node = "http://purl.obolibrary.org/obo/" + first_node
if second_node.startswith('GO'):
    second_node = "http://purl.obolibrary.org/obo/" + second_node

edges = pd.read_csv(input_file, header=None, sep='\t')
edges.columns = ["head", "rel", "tail"]

G = nx.Graph()
heads = edges["head"].tolist()
tails = edges["tail"].tolist()
G.add_edges_from(zip(heads, tails))

path = nx.shortest_path(G, source=first_node, target=second_node)
print(path)
