import numpy as np
import os
import pandas as pd
from src.cat import CatModel
import torch.optim as optim
import torch as th
import torch.nn as nn
from tqdm import tqdm, trange
import logging
from mowl.utils.data import FastTensorDataLoader
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CatMembership(CatModel):
    def __init__(self,
                 *args,
                 **kwargs):
        self._ontology_individuals = None
        self._ontology_individuals_idxs = None
        super().__init__(*args, **kwargs)

        self._node_to_id = None
        
        
    @property
    def classes_path(self):
        path = os.path.join(self.root, "classes.tsv")
        assert os.path.exists(path), f"Classes file {path} does not exist"
        return path

                                
    @property
    def individuals_path(self):
        path = os.path.join(self.root, "individuals.tsv")
        assert os.path.exists(path), f"Individuals file {path} does not exist"
        return path


    @property
    def ontology_individuals(self):
        if self._ontology_individuals is not None:
            return self._ontology_individuals
        
        individuals = pd.read_csv(self.individuals_path, sep=",", header=None)
        individuals.columns = ["individuals"]
        individuals = individuals["individuals"].values.tolist()
        individuals = set(individuals)
        bot = "owl:Nothing"
        top = "owl:Thing"
        individuals.add(bot)
        individuals.add(top)
        individuals = list(individuals)
        individuals.sort()

        self._ontology_individuals = individuals
        return self._ontology_individuals
    
    @property
    def ontology_individuals_idxs(self):
        if self._ontology_individuals_idxs is not None:
            return self._ontology_individuals_idxs
        
        class_to_id = {c: self.node_to_id[c] for c in self.ontology_individuals}
        ontology_individuals_idxs = th.tensor(list(class_to_id.values()), dtype=th.long, device=self.device)
        self._ontology_individuals_idxs = ontology_individuals_idxs
        return self._ontology_individuals_idxs


    @property
    def node_to_id(self):
        if self._node_to_id is not None:
            return self._node_to_id

        train_graph = pd.read_csv(self.graph_path, sep="\t", header=None)
        train_graph.columns = ["head", "rel", "tail"]

        valid_graph = pd.read_csv(self.validation_tuples_path, sep="\t", header=None)
        valid_graph.columns = ["head", "rel", "tail"]

        test_graph = pd.read_csv(self.test_tuples_path, sep="\t", header=None)
        test_graph.columns = ["head", "rel", "tail"]
        
        graph_classes = set(train_graph["head"].unique()) | set(train_graph["tail"].unique())
        graph_classes |= set(valid_graph["head"].unique()) | set(valid_graph["tail"].unique())
        graph_classes |= set(test_graph["head"].unique()) | set(test_graph["tail"].unique())
        
        # bot = bot_name[self.graph_type]
        # top = top_name[self.graph_type]
        # graph_classes.add(bot)
        # graph_classes.add(top)
                
        ont_classes = set(self.ontology_classes)
        all_classes = list(graph_classes | ont_classes | set(self.ontology_individuals)) 
        all_classes.sort()
        self._node_to_id = {c: i for i, c in enumerate(all_classes)}
        logger.info(f"Number of graph nodes: {len(self._node_to_id)}")
        return self._node_to_id


    def get_filtering_labels(self):
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.ontology_individuals_idxs)
        num_testing_tails = len(self.ontology_classes_idxs)
        
        filtering_labels = np.ones((num_testing_heads, num_testing_tails), dtype=np.int32)

        logging.debug(f"filtering_labels.shape: {filtering_labels.shape}")
                
        all_head_idxs = self.ontology_individuals_idxs.to(self.device)
        all_tail_idxs = self.ontology_classes_idxs.to(self.device)
        eval_rel_idx = None

        testing_dataloader = self.create_subsumption_dataloader(self.graph_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    head_ont_id = th.where(self.ontology_individuals_idxs == head_graph_id)[0]
                    rel = rel_idxs[i]
                    tail_graph_id = tail_idxs[i]
                    tail_ont_id = th.where(self.ontology_classes_idxs == tail_graph_id)[0]
                    filtering_labels[head_ont_id, tail_ont_id] = 10000
                    
        return filtering_labels

    def create_subsumption_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep="\t", header=None)
        num_cols = tuples.shape[1]

                
        if num_cols == 2:
            tuples.columns = ["head", "tail"]
        elif num_cols == 3:
            tuples.columns = ["head", "relation", "tail"]
        else:
            raise ValueError(f"Invalid number of columns in {tuples_path}")

        tuples = tuples.drop_duplicates()

        bot_iri = "http://www.w3.org/2002/07/owl#Nothing"
        top_iri = "http://www.w3.org/2002/07/owl#Thing"
        tuples["head"] = tuples["head"].apply(lambda x: 'owl:Nothing' if x == bot_iri else x)
        tuples["head"] = tuples["head"].apply(lambda x: 'owl:Thing' if x == top_iri else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: 'owl:Nothing' if x == bot_iri else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: 'owl:Thing' if x == top_iri else x)

        heads = []
        tails = []
        for h, t in zip(tuples["head"], tuples["tail"]):
            if h in self.ontology_individuals and t in self.ontology_classes:
                heads.append(self.node_to_id[h])
                tails.append(self.node_to_id[t])
                
                

        heads = th.tensor(heads, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)
        
        
        rel_idx = self.relation_to_id["http://arrow"]
        rels = rel_idx * th.ones_like(heads)
                                
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader






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
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics...", leave=False):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs)
                
                for i, graph_head in enumerate(head_idxs):
                    head = th.where(self.ontology_individuals_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.ontology_classes_idxs == graph_tail)[0]
                    
                    preds = predictions[i]
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
                                                                                    
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
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
        with tqdm(total=self.epochs, desc=f"Training. Best MRR: {best_mrr:.6f}. Best MR: {int(best_mr)}") as pbar:
        
            for epoch in range(self.epochs):
                pbar.set_description(f"Training. Best MRR: {best_mrr:.6f}. Best MR: {int(best_mr)}")
                pbar.update(1)
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


                    # batch_loss += self.model.collect_regularization_term().mean()


                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    graph_loss += batch_loss.item()


                graph_loss /= len(graph_dataloader)

                valid_every = 100
                if self.able_to_validate and epoch % valid_every == 0:
                    valid_mean_rank, valid_mrr = self.compute_ranking_metrics(mode="validate")
                    if valid_mrr > best_mrr:
                        best_mrr = valid_mrr
                        best_mr = valid_mean_rank
                        th.save(self.model.state_dict(), self.model_path)
                        tolerance = self.initial_tolerance+1
                        
                    else:
                        if valid_mean_rank < best_mr:
                            best_mr = valid_mean_rank
                            tolerance = self.initial_tolerance+1
                        else:
                            tolerance -= 1

                        


                    print(f"Training loss: {graph_loss:.6f}\tValidation mean rank: {valid_mean_rank:.6f}\tValidation MRR: {valid_mrr:.6f}")
                    wandb_logger.log({"epoch": epoch, "train_loss": graph_loss, "valid_mr": valid_mean_rank, "valid_mrr": valid_mrr})


                if tolerance == 0:
                    print("Early stopping")
                    break


    
    def test(self):
        logging.info("Testing ontology membership...")
        filtering_labels = self.get_filtering_labels()
        raw_metrics, filtered_metrics = self.compute_ranking_metrics(filtering_labels)
        return raw_metrics, filtered_metrics

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

