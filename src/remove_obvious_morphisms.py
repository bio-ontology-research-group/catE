import sys
import pandas as pd

input_file = sys.argv[1]
if not input_file.endswith('.edgelist'):
    raise ValueError('Input file must be an edgelist')

output_file = input_file.replace('.edgelist', '_filtered.edgelist')

df = pd.read_csv(input_file, sep='\t', header=None, names=['head', "rel", "tail"])


df = df[~((df['head'] == 'owl:Nothing') & (df['tail'].str.contains(' ')))]
df = df[~((df['tail'] == 'owl:Thing') & (df['head'].str.contains(' ')))]

df.to_csv(output_file, sep='\t', header=False, index=False)
