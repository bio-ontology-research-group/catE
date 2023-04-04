import sys
import pandas as pd
import networkx as nx

input_file = sys.argv[1]
if not input_file.endswith(".edgelist"):
    raise ValueError("Input file must be an edgelist")

edges = pd.read_csv(input_file, sep="\t", header=None)
edges.columns = ["head", "rel", "tail"]

G = nx.Graph()
heads = edges["head"].tolist()
tails = edges["tail"].tolist()
G.add_edges_from(zip(heads, tails))

components = list(nx.connected_components(G))
print("Number of connected components:", len(components))



