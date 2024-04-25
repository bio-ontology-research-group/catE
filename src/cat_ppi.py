from src.cat import CatModel
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT
import pandas as pd
import os
from scipy.stats import rankdata
from mowl.utils.data import FastTensorDataLoader
from mowl.datasets.builtin import PPIYeastDataset
from mowl.datasets import Dataset, OWLClasses

SUBSUMPTION_RELATION = "http://arrow"

class CatPPI(CatModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load()
        self._protein_names = None
        self._protein_idxs = None
        #self._existential_protein_idxs = None 

        print(f"Number of proteins: {len(self.protein_idxs)}")
        #print(f"Number of existential proteins: {len(self.existential_protein_idxs)}")

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        #graph_name = "ontology_extended.cat.edgelist" 
        graph_name = "ontology.cat.edgelist"
        
        graph_path = os.path.join(self.root, f"{graph_name}")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        print("Graph path", graph_path)
        return self._graph_path


    def _load(self):
        ds = PPIYeastDataset()
        classes = ds.classes.as_str
        
        ds = Dataset(ds.ontology)
        
        
        
        proteins = set()
        for owl_name, owl_cls in ds.classes.as_dict.items():
            if "http://4932" in owl_name:
                proteins.add(owl_cls)
        proteins = OWLClasses(proteins)
        proteins = proteins.as_str

        self._ontology_classes = classes
        self._protein_names = proteins
        
    @property
    def train_proteins_path(self):
        return os.path.join(self.root, "train_proteins.tsv")


    @property
    def ontology_classes(self):
        if self._ontology_classes is None:
            self._load()
                                                                                 
        return self._ontology_classes
    

    
    @property
    def protein_names(self):
        if self._protein_names is None:
            self._load()

        return self._protein_names
        
    @property
    def protein_idxs(self):
        if not self._protein_idxs is None:
            return self._protein_idxs

        
        protein_idxs = th.tensor([self.node_to_id[p] for p in self.protein_names], dtype=th.long, device=self.device)

        self._protein_idxs = protein_idxs
        return self._protein_idxs
        
    @property
    def existential_protein_idxs(self):
        if not self._existential_protein_idxs is None:
            return self._existential_protein_idxs

        existentials = [self.get_existential_node(p) for p in self.protein_names]
        ex_protein_idxs = th.tensor([self.node_to_id[p] for p in existentials], dtype=th.long, device=self.device)

        self._existential_protein_idxs = ex_protein_idxs
        return self._existential_protein_idxs


    
    @property
    def training_interactions_path(self):
        filename = f"train.tsv"
        path = os.path.join(self.root, filename)
        return path

    def get_existential_node(self, node):
        rel = "http://interacts_with"
        return f"{rel} some {node}"
        #return f"DOMAIN_{rel}_under_{rel} some {node}"
    
    def create_subsumption_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep="\t", header=None)
        tuples.columns = ["head", "tail"]

        rel_name = "http://interacts_with"

        heads = ["http://"+ h for h in tuples["head"].values]
        tails = ["http://"+ t for t in tuples["tail"].values]

        pairs = [(h, t) for h, t in zip(heads, tails) if h in self.protein_names and t in self.protein_names]

        heads, tails = zip(*pairs)
                                         
        heads = [self.node_to_id[h] for h in heads]
        tails = [self.node_to_id[self.get_existential_node(t)] for t in tails]

        heads = th.tensor(heads, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)

        morphism_id = self.relation_to_id["http://arrow"]
        rels = morphism_id * th.ones_like(heads)
                                                                            
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

        
    def get_filtering_labels(self):
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.protein_idxs)
        num_testing_tails = len(self.existential_protein_idxs)
        assert num_testing_heads == num_testing_tails, "Heads and tails should be the same size"
        
        filtering_labels = np.ones((num_testing_heads, num_testing_tails), dtype=np.int32)

        logging.debug(f"filtering_labels.shape: {filtering_labels.shape}")
                
        all_head_idxs = self.protein_idxs.to(self.device)
        all_tail_idxs = self.existential_protein_idxs.to(self.device)
        eval_rel_idx = None

        testing_dataloader = self.create_subsumption_dataloader(self.training_interactions_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    head_ont_id = th.where(self.protein_idxs == head_graph_id)[0]
                    rel = rel_idxs[i]
                    tail_graph_id = tail_idxs[i]
                    tail_ont_id = th.where(self.existential_protein_idxs == tail_graph_id)[0]
                    filtering_labels[head_ont_id, tail_ont_id] = 10000
                    
        return filtering_labels


    def compute_ranking_metrics(self, filtering_labels = None, mode = "test"):
        if not mode in ["test", "validate"]:
            raise ValueError(f"Mode {mode} is not valid")

        if filtering_labels is None and mode == "test":
            raise ValueError("Filtering labels should be provided for test mode")

        if filtering_labels is not None and mode == "validate":
            raise ValueError("Filtering labels should not be provided for validate mode")

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
                    head = th.where(self.protein_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.existential_protein_idxs == graph_tail)[0]

                    preds = predictions[i]


                    preds = preds.cpu().numpy()
                    rank = rankdata(-preds, method='average')[tail]
                    
                    # orderings = th.argsort(preds, descending=True)
                    # rank = th.where(orderings == tail)[0].item() + 1
                    mean_rank += rank
                    mrr += 1/rank
                    
                    if mode == "test":
                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1

                        filt_labels = filtering_labels[head, :]
                        filt_labels[tail] = 1
                        # filtered_preds = preds.cpu().numpy() * filt_labels
                        # filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_preds = preds * filt_labels
                        filtered_rank = rankdata(-filtered_preds, method='average')[tail]
                        # filtered_orderings = th.argsort(filtered_preds, descending=True) 
                        # filtered_rank = th.where(filtered_orderings == tail)[0].item() + 1
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
        # logits = self.model.distance(data)
        logits = logits.reshape(-1, len(self.protein_idxs))
        return logits

    def predict(self, heads, rels, tails):

        aux = heads.to(self.device)
        num_heads = len(heads)

        heads = heads.to(self.device)
        heads = heads.repeat(len(self.protein_idxs), 1).T
        assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
        heads = heads.reshape(-1)
        assert (aux[0] == heads[:num_heads]).all(), "heads are not the same"
        rels = rels.to(self.device)
        rels = rels.repeat(len(self.protein_idxs),1).T
        rels = rels.reshape(-1)
                                                
        eval_tails = self.existential_protein_idxs.repeat(num_heads)

        logits = self.normal_forward(heads, rels, eval_tails)

        return logits
        
    def test(self):
        logging.info("Testing ppi...")
        filtering_labels = self.get_filtering_labels()
        raw_metrics, filtered_metrics = self.compute_ranking_metrics(filtering_labels)
        return raw_metrics, filtered_metrics

    def compute_rank_roc(self, ranks):
        n_tails = len(self.existential_protein_idxs)
                    
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
