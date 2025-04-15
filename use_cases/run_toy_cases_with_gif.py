import sys
sys.path.append('../')
sys.path.append("../../")
sys.path.append("../../../")

import mowl
mowl.init_jvm("10g")
import click as ck
import os
from src.cat_toy_with_gif import CatToyWithGif
from mowl.utils.random import seed_everything
import gc
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
import imageio
from pathlib import Path
import glob

@ck.command()
@ck.option('--use_case', '-case', required=True, type=ck.Choice(["example5", "example5.1"]), default="example5")
@ck.option('--emb_dim', '-dim', required=True, type=int, default=2)
@ck.option('--batch_size', '-bs', required=True, type=int, default=1)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num_negs', '-negs', required=True, type=int, default=2)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--loss_type', '-loss', required=True, type=ck.Choice(["bpr", "normal"]), default="normal")
@ck.option('--epochs', '-e', required=True, type=int, default=1000)
@ck.option('--device', '-d', default="cuda")
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--not_sweep", '-ns', is_flag=True)
@ck.option("--description", '-desc', default="toy_case")
@ck.option("--save_every", '-se', type=int, default=10, help="Save embeddings every N epochs")
@ck.option("--gif_duration", '-gd', type=float, default=0.5, help="Duration of each frame in the GIF (seconds)")
def main(use_case, emb_dim, batch_size, lr, num_negs, margin,
         loss_type, epochs, device, seed, only_train,
         only_test, not_sweep, description, save_every, gif_duration):


    wandb_logger = wandb.init(project="cate2", name=description, group=f"example")

    if not_sweep:
        wandb_logger.log({"emb_dim": emb_dim,
                          "batch_size": batch_size,
                          "lr": lr,
                          "num_negs": num_negs,
                          "margin": margin,
                          "loss_type": loss_type,
                          })
    else:
        emb_dim = wandb.config.emb_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_negs = wandb.config.num_negs
        margin = wandb.config.margin
        loss_type = wandb.config.loss_type


    dir_ = use_case.split(".")[0]
    root = f"../tests/{dir_}/fixtures/"
    root_parent = os.path.dirname(root)
    
    
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    
    # Create a directory for GIF frames
    gif_dir = os.path.join(root_parent, 'gif_frames', use_case)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\temb_dim: ", emb_dim)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tnum_negs: ", num_negs)
    print("\tmargin: ", margin)
    print("\tloss_type: ", loss_type)
    print("\tepochs: ", epochs)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\tsave_every: ", save_every)
    seed_everything(seed)

    model = CatToyWithGif(use_case,
                          root,
                          emb_dim,
                          batch_size,
                          lr,
                          num_negs,
                          margin,
                          loss_type,
                          1,
                          epochs,
                          device,
                          seed,
                          10, #tolerance,
                          gif_dir,
                          save_every,
                          )

    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        kge_model = model.model
        triples_factory = model.triples_factory
        entity_to_id = triples_factory.entity_to_id
        kge_model.load_state_dict(th.load(model.model_path, weights_only=True))
        embeddings = model.model.entity_representations[0](indices=None)

        label_to_embedding = {label: embeddings[idx].detach().cpu().numpy() for label, idx in entity_to_id.items()}

        # Plot final embeddings
        plot_embeddings(label_to_embedding, model.graph, use_case)
        
        # Create GIF from saved frames
        create_gif(gif_dir, use_case, gif_duration)
            
def plot_embeddings(data, graph, use_case):
    heads = graph['head']
    tails = graph['tail']
    edges = zip(heads, tails)
            
    # Extract labels and coordinates
    labels = list(data.keys())
    points = np.array(list(data.values()))

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c="#1E5697", marker="o", s=100, edgecolors="#224E81", alpha=0.7)

    # Add labels to points
    for label, (x, y) in zip(labels, points):
        ha="right" if label in ["owl:Nothing"] else "left"
        plt.text(x + 0.01, y + 0.01, label, fontsize=9, ha=ha, va="bottom", color="red")


    plotted = {label: False for label in labels}
    points_to_scatter = set()
    # Draw arrows for edges
    for head, tail in edges:
        x_start, y_start = data[head]
        x_end, y_end = data[tail]
    
        # Draw an arrow
        if head not in ["A", "B"] and tail not in ["A", "B"]:
            alpha = 0.3
        else:
            alpha = 0.8

        dist_x = 0.95 * (x_end - x_start)
        dist_y = 0.95 * (y_end - y_start)
        plt.arrow(x_start, y_start, dist_x, dist_y,
                  head_width=0.01, length_includes_head=True, color="#9E5959", alpha=alpha)

    # Formatting the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    if use_case == "example5":
        plt.title("Embedding for theory T (Final)")
    elif use_case == "example5.1":
        plt.title("Embedding for theory T' (Final)")
    plt.grid(True)
    plt.savefig(f"/home/zhapacfp/data/{use_case}_final.png", dpi=300)
    plt.close()

def create_gif(gif_dir, use_case, duration=0.5):
    """Create a GIF from the saved embedding snapshots"""
    # Get all PNG files in the directory, sorted by epoch number
    png_files = sorted(glob.glob(os.path.join(gif_dir, f"{use_case}_epoch_*.png")))
    
    if not png_files:
        print(f"No PNG files found in {gif_dir} for {use_case}")
        return
    
    print(f"Creating GIF from {len(png_files)} frames...")
    
    # Create GIF
    output_path = f"/home/zhapacfp/data/{use_case}_training.gif"
    
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)
    
    print(f"GIF saved to {output_path}")
    
if __name__ == "__main__":
    main()
