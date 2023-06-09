import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
import os
from tqdm import tqdm

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    
    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    outfile_name = os.path.split(input_ontology)[0] + "/properties.txt"

    properties = ontology.getObjectPropertiesInSignature()
    with open(outfile_name, "w") as outfile:
        for rel in tqdm(properties):
            rel_str = str(rel.toStringID())
            outfile.write(f"{rel_str}\n")
    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
