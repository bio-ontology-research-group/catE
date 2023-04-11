import pandas as pd
import sys


subclass_name = {
    "onto2graph": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "rdf": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "owl2vec": "http://subclassof",
    "cat": "http://arrow"
}

edges_file = sys.argv[1]
if not edges_file.endswith('.edgelist'):
    raise ValueError('Input file must be an edgelist')

test_file = sys.argv[2]
if not test_file.endswith('.csv'):
    raise ValueError('Test file must be a csv')

graph_type = sys.argv[3]
if not graph_type in subclass_name:
    raise ValueError('Graph type must be one of {}'.format(subclass_name.keys()))

edges_df = pd.read_csv(edges_file, sep='\t', header=None)
edges_df.columns = ["head", "relation", "tail"]
print('Number of edges before removing duplicates: {}'.format(len(edges_df)))
print("Removing duplicates...")
edges_df.drop_duplicates(inplace=True)
print('Number of edges after removing duplicates: {}'.format(len(edges_df)))

test_df = pd.read_csv(test_file, sep=',', header=None)
test_df.columns = ["head", "tail"]

subclass_df = edges_df[edges_df['relation'] == subclass_name[graph_type]]
subclass_df = subclass_df[["head", "tail"]]
print(f"Number of subclass edges: {len(subclass_df)}")


leakage = test_df.merge(subclass_df, how="inner").drop_duplicates()
print("Number of training edges in test set: {}".format(len(leakage)))
print("Removing leakage...")

#add subclass as a new column
leakage['relation'] = subclass_name[graph_type]
leakage = leakage[["head", "relation", "tail"]]

new_train_df = edges_df.merge(leakage, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
print("New test edges:", len(new_train_df))
out_file = edges_file.replace(".edgelist", "_no_leakage.edgelist")
new_train_df.to_csv(out_file, sep="\t", header=False, index=False)
print("New train edges written to:", out_file)
leakage.to_csv(f"leakage_{graph_type}.csv", sep="\t", header=False, index=False)
print("Leakage written to:", f"leakage_{graph_type}.csv")
