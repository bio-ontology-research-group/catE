import sys
import os
import pandas as pd

root = sys.argv[1]
prefix = sys.argv[2]
owl2vec = os.path.join(root, prefix + '.owl2vec.edgelist')
onto2graph = os.path.join(root, prefix + '.onto2graph.edgelist')
rdf = os.path.join(root, prefix + '.rdf.edgelist')
cat = os.path.join(root, prefix + '.cat.edgelist')
cat1 = os.path.join(root, prefix + '.cat.s1.edgelist')
test_file = os.path.join(root, "text.csv")
assert os.path.exists(owl2vec)
assert os.path.exists(onto2graph)
assert os.path.exists(rdf)
assert os.path.exists(cat)
assert os.path.exists(cat1)

df_owl2vec = pd.read_csv(owl2vec, sep='\t', header=None)
df_onto2graph = pd.read_csv(onto2graph, sep='\t', header=None)
df_rdf = pd.read_csv(rdf, sep='\t', header=None)
df_cat = pd.read_csv(cat, sep='\t', header=None)
df_cat1 = pd.read_csv(cat1, sep='\t', header=None)

df_owl2vec.columns = ['head', "rel", 'tail']
df_onto2graph.columns = ['head', "rel", 'tail']
df_rdf.columns = ['head', "rel", 'tail']
df_cat.columns = ['head', "rel", 'tail']
df_cat1.columns = ['head', "rel", 'tail']

nodes_
