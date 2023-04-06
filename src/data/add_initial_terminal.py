import mowl
mowl.init_jvm("10g")

import sys
sys.path.append("../../")

import pandas as pd
from mowl.owlapi.defaults import BOT, TOP
from src.utils import subsumption_rel_name, bot_name, top_name
import click as ck
from tqdm  import tqdm

import logging
logging.basicConfig(level=logging.INFO)

def add_initial_and_terminal(input_file, classes_file, projector_name):
    if not input_file.endswith(".edgelist"):
        raise ValueError("Input file must be an edgelist")
    if not classes_file.endswith(".txt"):
        raise ValueError("Classes file must be a txt file")
    
    df = pd.read_csv(input_file, sep='\t')
    df.columns = ["head", "relation", "tail"]

    logging.info(f"Number of rows: {len(df)}")
    df.drop_duplicates(inplace=True)
    logging.info(f"Number of rows after dropping duplicates: {len(df)}")

    classes = pd.read_csv(classes_file, sep='\t')
    classes.columns = ["class"]
    classes_set = set(classes["class"].tolist())
    logging.info(f"Number of ontology classes: {len(classes_set)}")

    sub_rel_name = subsumption_rel_name[projector_name]
    
    added = 0
    bot = bot_name[projector_name]
    top = top_name[projector_name]

    new_edges = []
    for cls in tqdm(classes_set, total=len(classes_set), desc="Adding initial and terminal nodes"):
        new_edges.append((bot, sub_rel_name, cls))
        new_edges.append((cls, sub_rel_name, top))
        added += 2

    df = pd.concat([df, pd.DataFrame(new_edges, columns=["head", "relation", "tail"])])
    
    logging.info(f"No. of rows added: {added}")
    logging.info(f"Total no. of rows: {len(df)}")
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Total no. of rows after dropping duplicates: {len(df)}")
    
    output_file = input_file.replace(".edgelist", "_initial_terminal.edgelist")
    df.to_csv(output_file, sep='\t', index=False, header=False)
    print("Saved to", output_file)

@ck.command()
@ck.option('--input_file', '-i', type=ck.Path(exists=True), required=True)
@ck.option('--classes_file', '-c', type=ck.Path(exists=True), required=True)
@ck.option('--projector_name', '-p', type=ck.Choice(["owl2vec", "onto2graph", "rdf", "cat", "cat1"]), required=True)
def main(input_file, classes_file, projector_name):
    add_initial_and_terminal(input_file, classes_file, projector_name)


if __name__ == "__main__":
    main()
