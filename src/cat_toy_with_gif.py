from src.cat import CatModel, OrderE
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT
import pandas as pd
import os
from mowl.utils.data import FastTensorDataLoader
import matplotlib.pyplot as plt

SUBSUMPTION_RELATION = "http://arrow"

class CatToyWithGif(CatModel):
    def __init__(self,
                 use_case,
                 root,
                 emb_dim,
                 batch_size,
                 lr,
                 num_negs,
                 margin,
                 loss_type,
                 p,
                 epochs,
                 device,
                 seed,
                 initial_tolerance,
                 gif_dir,
                 save_every=10,
                 ):

        self.use_case = use_case
        self.root = root
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_negs = num_negs
        self.margin = margin
        self.loss_type = loss_type
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.initial_tolerance = initial_tolerance
        self._triples_factory = None
        self._graph = None
        self._graph_path = None
        self._model_path = None
        self._node_to_id = None
        self._relation_to_id = None
        self._id_to_node = None
        self._id_to_relation = None
        self._ontology_classes = None
        self._ontology_properties = None
        self._ontology_classes_idxs = None
        self._ontology_properties_idxs = None
        self._ancestors_path = None
        
        # GIF related parameters
        self.gif_dir = gif_dir
        self.save_every = save_every
        self.embedding_snapshots = []
        
        self.able_to_validate = False
        
        print("Parameters:")
        print(f"\tUse case: {self.use_case}")
        print(f"\tRoot: {self.root}")
        print(f"\tEmbedding dimension: {self.emb_dim}")
        print(f"\tBatch size: {self.batch_size}")
        print(f"\tLearning rate: {self.lr}")
        print(f"\tNumber of negatives: {self.num_negs}")
        print(f"\tEpochs: {self.epochs}")
        print(f"\tDevice: {self.device}")
        print(f"\tSeed: {self.seed}")
        print(f"\tGIF directory: {self.gif_dir}")
        print(f"\tSave embeddings every: {self.save_every} epochs")
        
        self.model = OrderE(triples_factory=self.triples_factory,
                            embedding_dim=self.emb_dim,
                            random_seed=self.seed, p=p)

        assert os.path.exists(self.root), f"Root directory {self.root} does not exist."
        
        # Create GIF directory if it doesn't exist
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        
    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        
        print(f"Loading graph from {self.graph_path}")
        graph = pd.read_csv(self.graph_path, sep=",", header=None)
        graph.columns = ["head", "relation", "tail"]
        self._graph = graph
        print("Done")
        return self._graph

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path
        if 'example5.1' in self.use_case:
            graph_name = "example5.1.cat.edgelist"
        elif 'example5' in self.use_case:
            graph_name = "example5.cat.edgelist"
        else:
            raise ValueError(f"Unknown use case {self.use_case}")

        graph_path = os.path.join(self.root, graph_name)
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        return self._graph_path

    @property
    def model_path(self):
        if self._model_path is not None:
            return self._model_path

        params_str = f"{self.use_case}"
        params_str += f"_dim{self.emb_dim}"
        params_str += f"_bs{self.batch_size}"
        params_str += f"_lr{self.lr}"
        params_str += f"_negs{self.num_negs}"
        params_str += f"_margin{self.margin}"
        params_str += f"_loss{self.loss_type}"
        
        models_dir = os.path.dirname(self.root)
        models_dir = os.path.join(models_dir, "models")

        basename = f"{params_str}.model.pt"
        self._model_path = os.path.join(models_dir, basename)
        return self._model_path

    def save_embedding_snapshot(self, epoch):
        """Save current embeddings for GIF creation"""
        embeddings = self.model.entity_representations[0](indices=None).detach().cpu().numpy()
        entity_to_id = self.triples_factory.entity_to_id
        label_to_embedding = {label: embeddings[idx] for label, idx in entity_to_id.items()}
        self.embedding_snapshots.append((epoch, label_to_embedding))
        
        # Also save as an image for debugging
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
            plt.title(f"Embedding for theory T - Epoch {epoch}")
        elif use_case == "example5.1":
            plt.title(f"Embedding for theory T' - Epoch {epoch}")
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(self.gif_dir, f"{use_case}_epoch_{epoch:04d}.png"), dpi=150)
        plt.close()
    
    def compute_ranking_metrics(self, filtering_labels=None, mode="test"):
        if filtering_labels is None and mode == "test":
            raise ValueError("filtering_labels cannot be None when mode is test")

        if filtering_labels is not None and mode == "validate":
            raise ValueError("filtering_labels must be None when mode is validate")

        if mode == "test":
            print(f"Loading best model from {self.model_path}")
            self.model.load_state_dict(th.load(self.model_path))
            self.model = self.model.to(self.device)

        self.model.eval()
        mean_rank, filtered_mean_rank = 0, 0
        mrr, filtered_mrr = 0, 0

        if mode == "test":
            hits_at_k = dict([(k, 0) for k in [1, 3, 10, 50, 100]])
            fhits_at_k = dict([(k, 0) for k in [1, 3, 10, 50, 100]])
            ranks, filtered_ranks = dict(), dict()

        tuples_path = self.test_tuples_path if mode == "test" else self.validation_tuples_path

        testing_dataloader = self.create_subsumption_dataloader(tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics..."):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs)
                
                for i, graph_head in enumerate(head_idxs):
                    head = th.where(self.ontology_classes_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.ontology_classes_idxs == graph_tail)[0]
                    
                    preds = predictions[i]
                    preds[head] = -10000
                    orderings = th.argsort(preds, descending=True)
                    rank = th.where(orderings == tail)[0].item() + 1
                    mean_rank += rank
                    mrr += 1/(rank)
                    
                    
                    if mode == "test":
                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1

                        filt_labels = filtering_labels[head, :]
                        filt_labels[tail] = 1
                        filtered_preds = preds.cpu().numpy() * filt_labels
                        filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_orderings = th.argsort(filtered_preds, descending=True) 
                        filtered_rank = th.where(filtered_orderings == tail)[0].item() + 1
                        filtered_mean_rank += filtered_rank
                        filtered_mrr += 1/(filtered_rank)

                        for k in hits_at_k.keys():
                            if rank < k:
                                hits_at_k[k] += 1
                            if filtered_rank < k:
                                fhits_at_k[k] += 1
                        

                        if filtered_rank not in filtered_ranks:
                            filtered_ranks[filtered_rank] = 0
                        filtered_ranks[filtered_rank] += 1

            mean_rank /= testing_dataloader.dataset_len
            mrr /= testing_dataloader.dataset_len
            
            if mode == "test":
                for k in hits_at_k.keys():
                    hits_at_k[k] /= testing_dataloader.dataset_len
                    fhits_at_k[k] /= testing_dataloader.dataset_len
                
                auc = self.compute_rank_roc(ranks)

                filtered_mean_rank /= testing_dataloader.dataset_len
                filtered_mrr /= testing_dataloader.dataset_len
                fauc = self.compute_rank_roc(filtered_ranks)

                raw_metrics = {"mr": mean_rank, "mrr": mrr, "auc": auc}
                filtered_metrics = {"fmr": filtered_mean_rank, "fmrr": filtered_mrr, "fauc": fauc}

                for k in hits_at_k.keys():
                    raw_metrics[f"hits_{k}"] = hits_at_k[k]
                    filtered_metrics[f"fhits_{k}"] = fhits_at_k[k]
                
                return raw_metrics, filtered_metrics
            else:
                return mean_rank, mrr
            
        
                                                                                     
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
        data = th.vstack((head_idxs, rel_idxs, tail_idxs)).to(self.device)
        data = data.T
        
        logits = self.model.score_hrt(data)
        logits = logits.reshape(-1, len(self.ontology_classes_idxs))
        return logits

    def predict(self, heads, rels, tails):

        aux = heads.to(self.device)
        num_heads = len(heads)

        heads = heads.to(self.device)
        heads = heads.repeat(len(self.ontology_classes_idxs), 1).T
        assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
        heads = heads.reshape(-1)
        assert (aux[0] == heads[:num_heads]).all(), "heads are not the same"
        rels = rels.to(self.device)
        rels = rels.repeat(len(self.ontology_classes_idxs),1).T
        rels = rels.reshape(-1)
                                                
        eval_tails = self.ontology_classes_idxs.repeat(num_heads)

        logits = self.normal_forward(heads, rels, eval_tails)

        return logits
        

    def train(self, wandb_logger):
        
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        min_lr = self.lr/10
        max_lr = self.lr
        print("Min lr: {}, Max lr: {}".format(min_lr, max_lr))
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        graph_dataloader = self.create_graph_train_dataloader()

        tolerance = 0
        best_loss = float("inf")
        best_mr = 10000000
        best_mrr = 0
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)
        
        # Save initial embeddings
        self.save_embedding_snapshot(0)

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

            # Save embedding snapshot every save_every epochs
            if epoch % self.save_every == 0 or epoch == self.epochs - 1:
                self.save_embedding_snapshot(epoch + 1)  # +1 to make it 1-indexed

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
                # Save final embeddings if we're stopping early
                if (epoch + 1) % self.save_every != 0:
                    self.save_embedding_snapshot(epoch + 1)
                break


    def compute_rank_roc(self, ranks):
        n_tails = len(self.ontology_classes_idxs)
                    
        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())
        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)
        auc_x.append(n_tails)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x) / n_tails
        return auc
