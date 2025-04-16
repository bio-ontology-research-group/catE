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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import random
import wandb
import glob
import pickle

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
@ck.option("--save_every", '-se', default=2, type=int, help="Save embeddings every N epochs")
@ck.option("--embeddings_dir", '-ed', default="./embeddings", help="Directory to save embeddings")
def main(use_case, emb_dim, batch_size, lr, num_negs, margin,
         loss_type, epochs, device, seed, only_train,
         only_test, not_sweep, description, save_every, embeddings_dir):

    # Create a unique directory for this run
    embeddings_dir = os.path.join(embeddings_dir, f"{use_case}_{description}")
    os.makedirs(embeddings_dir, exist_ok=True)

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
    print("\tembeddings_dir: ", embeddings_dir)
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
                          embeddings_dir,
                          save_every
                          )

    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        # Create plots for all saved embeddings
        create_embedding_plots(model.graph, use_case, embeddings_dir)
            
def create_embedding_plots(graph, use_case, embeddings_dir):
    """Create plots for all saved embeddings"""
    # Get all embedding files
    embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, "step_*.pkl")))
    print(f"Found {len(embedding_files)} embedding files")
    
    # Create output directory for plots
    plots_dir = os.path.join(embeddings_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Process each file
    for file_path in embedding_files:
        # Extract epoch number from filename
        epoch = int(os.path.basename(file_path).replace("step_", "").replace(".pkl", ""))
        
        # Load embeddings
        with open(file_path, 'rb') as f:
            label_to_embedding = pickle.load(f)
        
        # Create plot
        plot_embeddings(label_to_embedding, graph, use_case, epoch, plots_dir)
    
    print(f"Created {len(embedding_files)} plots in {plots_dir}")
    duration=500
    delay=duration/len(embedding_files)
    print(f"To create a GIF that lasts {duration/100} seconds, you can use: convert -delay {delay} -loop 0 {plots_dir}/step_*.png animation.gif")

def plot_embeddings(data, graph, use_case, epoch, output_dir):
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
        plt.title(f"Embedding for theory T (Epoch {epoch})")
    elif use_case == "example5.1":
        plt.title(f"Embedding for theory T' (Epoch {epoch})")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"step_{epoch:04d}.png"), dpi=300)
    plt.close()
    
if __name__ == "__main__":
    main()
