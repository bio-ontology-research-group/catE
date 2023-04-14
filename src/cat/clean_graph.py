import mowl
mowl.init_jvm("10g")
import sys
sys.path.append("../../")
import pandas as pd
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from src.utils import bot_name
import time

graph_file = sys.argv[1]
if not graph_file.endswith('.edgelist'):
    raise ValueError('Edges file must be in .edgelist format')

bot_names = ["owl:Nothing", "not owl:Thing"]
top_names = ["owl:Thing", "not owl:Nothing"]


edges = pd.read_csv(graph_file, header=None, sep='\t')
edges.columns = ["head", "rel", "tail"]
edges = edges.drop_duplicates()

G = nx.DiGraph()
heads = edges["head"].tolist()
tails = edges["tail"].tolist()
G.add_edges_from(zip(heads, tails))


def find_paths2():
    print("Finding paths...")
    paths = []
    for top in top_names:
        for bot in bot_names:
            print("Finding paths from {} to {}".format(top, bot))
            start = time.time()
            all_paths = nx.all_simple_paths(G, source=top, target=bot)
            for path in all_paths:
                if len(path) > 1:
                    paths.append(path)
            end = time.time()
            print("Found {} paths in {} seconds".format(len(paths), end - start))
    return path

def remove_paths():
    print("Finding paths...")
    paths = []
    for top in top_names:
        for bot in bot_names:
            print("Finding and removing paths from {} to {}".format(top, bot))
            start = time.time()
            exist_shortest_path = True
            while exist_shortest_path:
                try:
                    shortest_path = nx.shortest_path(G, source=top, target=bot)
                    for i in range(len(shortest_path)-1):
                        G.remove_edge(shortest_path[i], shortest_path[i+1])
                except nx.NetworkXNoPath:
                    exist_shortest_path = False
                except nx.NodeNotFound:
                    exist_shortest_path = False
                        
            end = time.time()
            print("Removed paths in {} seconds".format(end - start))
    
remove_paths()

#rewrite_graph
new_edges = []
for edge in G.edges():
    new_edges.append([edge[0], "http://arrow", edge[1]])

new_edges = pd.DataFrame(new_edges)
         
outfile = graph_file.replace('.edgelist', '_cleaned.edgelist')
new_edges.to_csv(outfile, sep='\t', header=False, index=False)
