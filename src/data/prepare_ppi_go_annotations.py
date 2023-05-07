import os
import sys
import gzip


org_to_id = {'yeast': '4932', 'human': '9606'}

organism = sys.argv[1]
org_id = org_to_id[organism]


root = "../../use_cases/ppi/data/"
assert os.path.exists(root)

raw_data_path = os.path.join(root, "raw_data")
assert os.path.exists(raw_data_path)



mapping = {}
source = {'4932': 'SGD_ID', '9606': 'Ensembl_UniProt_AC'} # mapping source

with gzip.open(f'{raw_data_path}/{org_id}.protein.aliases.v11.0.txt.gz', 'rt') as f:
    next(f) # Skip header
    for line in f:
        string_id, p_id, sources = line.strip().split('\t')
        if source[org_id] not in sources.split():
            continue
        if p_id not in mapping:
            mapping[p_id] = set()
        mapping[p_id].add(string_id)
print('Loaded mappings', len(mapping))

gaf_files = {'4932': 'sgd.gaf.gz', '9606': 'goa_human.gaf.gz'}
annotations = set()
with gzip.open(f'{raw_data_path}/{gaf_files[org_id]}', 'rt') as f:
    for line in f:
        if line.startswith('!'): # Skip header
            continue
        it = line.strip().split('\t')
        p_id = it[1]
        go_id = it[4]
        if it[6] == 'IEA' or it[6] == 'ND': # Ignore predicted or no data annotations
            continue
        if p_id not in mapping: # Not in StringDB
            continue
        s_ids = mapping[p_id]
        for s_id in s_ids:
            annotations.add((s_id, go_id))
print('Number of annotations:', len(annotations))

# Save annotations
with open(f'{raw_data_path}/train/{org_id}.go.annotation.txt', 'w') as f:
    for p_id, go_id in annotations:
        f.write(f'{p_id}\t{go_id}\n')

