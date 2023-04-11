import sys
import pandas as pd

edges_file = sys.argv[1]
classes_file = sys.argv[2]

edges_df = pd.read_csv(edges_file, sep='\t', header=None)
edges_df.columns = ['head', "rel", "tail"]

classes_df = pd.read_csv(classes_file, sep='\t', header=None)
classes_df.columns = ['class']

nodes = set(edges_df['head'].tolist() + edges_df['tail'].tolist())
classes = set(classes_df['class'].tolist())

for cls in classes:
    if cls not in nodes:
        raise ValueError("Class {} is not in the nodes".format(cls))

print("All classes are in the nodes")
