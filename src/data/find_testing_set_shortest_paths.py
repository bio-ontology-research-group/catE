import sys
sys.path.append("../")
import pandas as pd
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from utils import bot_name

edges_file = sys.argv[1]
if not edges_file.endswith('.edgelist'):
    raise ValueError('Edges file must be in .edgelist format')
test_set_file = sys.argv[2]
if not (test_set_file.endswith('.csv') or  test_set_file.endswith('.txt')):
    raise ValueError('Test set file must be in .csv or .txt format')
graph_type = sys.argv[3]
if graph_type not in ["onto2graph", "owl2vec", "rdf", "cat", "cat1", "cat2"]:
    raise ValueError("Graph type not recognized")

bot = bot_name[graph_type]

edges = pd.read_csv(edges_file, header=None, sep='\t')
edges.columns = ["head", "rel", "tail"]
edges = edges.drop_duplicates()

if test_set_file.endswith('.csv'):
    test_set = pd.read_csv(test_set_file, header=None, sep=',')
    test_set.columns = ["head", "tail"]
    test_set["tail"] = test_set["tail"].apply(lambda x: bot if x == "owl:Nothing" else x)
else:
    test_set = pd.read_csv(test_set_file, header=None, sep='\t')
    test_set.columns = ["head"]
    test_set["tail"] = bot

G = nx.DiGraph()
heads = edges["head"].tolist()
tails = edges["tail"].tolist()
G.add_edges_from(zip(heads, tails))

with_path = 0
without_path = 0
node_not_found = 0
avg_path_length = 0
not_found_nodes = set()


def find_path(nodes):
    with_path, without_path, node_not_found, avg_path_length, not_found_nodes = 0, 0, 0, 0, set()
    first_node, second_node = nodes
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

    return with_path, without_path, node_not_found, avg_path_length, not_found_nodes

with mp.Pool(processes=16) as pool:
    max_ = len(test_set)
    nodes_pairs = zip(test_set["head"].tolist(), test_set["tail"].tolist())
    with tqdm(total=max_) as pbar:
        for r in pool.imap_unordered(find_path, nodes_pairs):
            with_path += r[0]
            without_path += r[1]
            node_not_found += r[2]
            avg_path_length += r[3]
            not_found_nodes.update(r[4])
            pbar.update()

avg_path_length = avg_path_length / (with_path)

info_file = edges_file.replace('.edgelist', '_info_testing_shortest_paths.txt')
with open(info_file, 'w') as f:
    f.write(f'With path: {with_path}\n')
    f.write(f'Without path: {without_path}\n')
    f.write(f'Node not found: {node_not_found}\n')
    f.write(f'Average path length: {avg_path_length}\n')
    f.write(f'Not found nodes: {not_found_nodes}')
                
print("With path: ", with_path)
print("Average path length: ", avg_path_length)

print("Without path: ", without_path)
print("Node not found: ", node_not_found, len(not_found_nodes))


