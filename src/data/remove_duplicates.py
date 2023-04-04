import pandas as pd
import sys

filename = sys.argv[1]
# Read the CSV file
df = pd.read_csv(filename, sep="\t")

# Drop duplicates
df.drop_duplicates(inplace=True)

if not "edgelist" in filename:
    raise ValueError("filename name is not correct")

outfile = filename.replace(".edgelist", ".no.duplicates.edgelist")
# Save the cleaned data to a new CSV file
df.to_csv(outfile, index=False, sep="\t")
