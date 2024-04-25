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
    outfile_name = os.path.split(input_ontology)[0] + "/classes.txt"

    classes = ontology.getClassesInSignature()
    with open(outfile_name, "w") as outfile:
        for cls in tqdm(classes):
            cls_str = str(cls.toStringID())
            outfile.write(f"{cls_str}\n")
    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
