import sys
import pandas as pd

input_file = sys.argv[1]
if not input_file.endswith('.edgelist'):
    raise ValueError('Input file must be an edgelist')

output_file = input_file.replace('.edgelist', '_no_initial_and_terminal.edgelist')

df = pd.read_csv(input_file, sep='\t', header=None, names=['head', "rel", "tail"])

df = df[~df['tail'].str.contains('owl:Nothing')]
df = df[~df['head'].str.contains('owl:Thing')]
df = df.drop_duplicates().dropna()

df.to_csv(output_file, sep='\t', header=False, index=False)
