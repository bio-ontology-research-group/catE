import mowl
mowl.init_jvm("10g")

import sys
sys.path.append("../../")

import pandas as pd
from mowl.owlapi.defaults import BOT, TOP
from src.utils import subsumption_rel_name, bot_name, top_name
import click as ck

import logging
logging.basicConfig(level=logging.INFO)

def add_initial_and_terminal(input_file, projector_name):
    df = pd.read_csv(input_file, sep='\t')
    df.columns = ["head", "relation", "tail"]

    logging.info(f"Number of rows: {len(df)}")
    df.drop_duplicates(inplace=True)
    logging.info(f"Number of rows after dropping duplicates: {len(df)}")
    
    nodes = set(df["head"].tolist() + df["tail"].tolist())
    logging.info(f"Number of nodes: {len(nodes)}")

    
    if not input_file.endswith(".edgelist"):
        raise ValueError("Input file must be an edgelist")

    sub_rel_name = subsumption_rel_name[projector_name]
    # Add initial and terminal nodes
    added = 0
    bot = bot_name[projector_name]
    top = top_name[projector_name]
    
    for node in nodes:
        df = pd.concat([df, pd.DataFrame([[node, sub_rel_name, top], [bot, sub_rel_name, node]], columns=["head", "relation", "tail"])])
        added += 2

    logging.info(f"No. of rows added: {added}")
    logging.info(f"Total no. of rows: {len(df)}")
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Total no. of rows after dropping duplicates: {len(df)}")
    
    output_file = input_file.replace(".edgelist", "_initial_terminal.edgelist")
    df.to_csv(output_file, sep='\t', index=False, header=False)
    print("Saved to", output_file)

@ck.command()
@ck.option('--input_file', '-i', type=ck.Path(exists=True), required=True)
@ck.option('--projector_name', '-p', type=ck.Choice(["owl2vec", "onto2graph", "rdf", "cat"]), required=True)
def main(input_file, projector_name):
    add_initial_and_terminal(input_file, projector_name)


if __name__ == "__main__":
    main()
