from src.cat_toy import CatToy
import torch as th
import os
import pickle
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

class CatToyWithGif(CatToy):
    def __init__(self, use_case, root, emb_dim, batch_size, lr, num_negs, margin, 
                 loss_type, p, epochs, device, seed, initial_tolerance, 
                 save_embeddings_dir='./embeddings', save_every=10):
        # Save GIF-related parameters
        self.save_embeddings_dir = save_embeddings_dir
        self.save_every = save_every
        
        # Create the directory if it doesn't exist
        os.makedirs(self.save_embeddings_dir, exist_ok=True)
        
        # Call the parent constructor
        super().__init__(use_case, root, emb_dim, batch_size, lr, num_negs, margin,
                         loss_type, p, epochs, device, seed, initial_tolerance)
        
    def save_current_embeddings(self, epoch):
        """Save the current embeddings to a pickle file and create a plot"""
        self.model.eval()
        with th.no_grad():
            embeddings = self.model.entity_representations[0](indices=None)
            label_to_embedding = {
                self.id_to_node[idx]: embeddings[idx].detach().cpu().numpy() 
                for idx in range(len(self.id_to_node))
            }
            
            # Save to a pickle file
            save_path = os.path.join(self.save_embeddings_dir, f"step_{epoch}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(label_to_embedding, f)
            
            # Also create and save a plot
            self.plot_embeddings(label_to_embedding, self.graph, self.use_case, epoch)
    
    def plot_embeddings(self, data, graph, use_case, epoch):
        """Plot and save current embeddings"""
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
        
        # Save the figure
        plots_dir = os.path.join(self.save_embeddings_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"step_{epoch:04d}.png"), dpi=150)
        plt.close()
                
    def train(self, wandb_logger):
        """Override the train method to save embeddings at each epoch"""
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        min_lr = self.lr/10
        max_lr = self.lr
        print("Min lr: {}, Max lr: {}".format(min_lr, max_lr))
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = th.nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        graph_dataloader = self.create_graph_train_dataloader()

        tolerance = 0
        best_loss = float("inf")
        best_mr = 10000000
        best_mrr = 0
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)

        # Save initial embeddings
        self.save_current_embeddings(0)

        for epoch in trange(self.epochs, desc=f"Training..."):
            # logging.info(f"Epoch: {epoch+1}")
            self.model.train()

            graph_loss = 0
            for head, rel, tail in graph_dataloader:
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)

                pos_logits = self.model.forward(head, rel, tail)

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                    neg_logits += self.model.forward(head, rel, neg_tail)
                neg_logits /= self.num_negs

 
                if self.loss_type == "bpr":
                    batch_loss = -criterion_bpr(self.margin + pos_logits).mean() - criterion_bpr(-neg_logits - self.margin).mean()
                elif self.loss_type == "normal":
                    batch_loss = -pos_logits.mean() + th.relu(self.margin + neg_logits).mean()
                    # batch_loss = (-pos_logits + th.relu(self.margin + neg_logits)).mean()

                    
                # batch_loss += self.model.collect_regularization_term().mean()
                
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()
                

            graph_loss /= len(graph_dataloader)

            # Save embeddings every save_every epochs
            if epoch % self.save_every == 0 or epoch == self.epochs - 1:
                self.save_current_embeddings(epoch + 1)  # +1 so we start at 1 not 0

            valid_every = 1
            if epoch % valid_every == 0:
                if graph_loss < best_loss:
                    best_loss = graph_loss
                    th.save(self.model.state_dict(), self.model_path)
                    tolerance = self.initial_tolerance+1
                    print("Model saved")
                else:
                    tolerance -= 1

                print(f"Training loss: {graph_loss:.6f}")
                wandb_logger.log({"epoch": epoch, "train_loss": graph_loss})
            
                                    
            if tolerance == 0:
                print("Early stopping")
                # Save final embeddings before breaking
                self.save_current_embeddings(epoch + 1)
                break
