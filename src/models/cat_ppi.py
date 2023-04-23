from src.models.cat import CatModel
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT
import os
import pandas as pd
from src.utils import FastTensorDataLoader, subsumption_rel_name, prefix, graph_type, suffix_ppi

class CatPPI(CatModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._protein_idxs = None
        self._existential_protein_idxs = None

        print(f"Number of proteins: {len(self.protein_idxs)}")
        print(f"Number of existential proteins: {len(self.existential_protein_idxs)}")

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        pref = "yeast-classes_extended"
        graph_name = graph_type[self.graph_type]
        suf = suffix_ppi[self.graph_type]
        graph_path = os.path.join(self.root, f"{pref}.{graph_name}{suf}")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        print("Graph path", graph_path)
        return self._graph_path

    @property
    def protein_idxs(self):
        if not self._protein_idxs is None:
            return self._protein_idxs
            
        protein_names = [p for p in self.ontology_classes if p.startswith("http://4932")]
        protein_idxs = th.tensor([self.node_to_id[p] for p in protein_names], dtype=th.long, device=self.device)

        self._protein_idxs = protein_idxs
        return self._protein_idxs
        
    @property
    def existential_protein_idxs(self):
        if not self._existential_protein_idxs is None:
            return self._existential_protein_idxs
        
        protein_names = [p for p in self.ontology_classes if p.startswith("http://4932")]
        existential_protein_names = [self.get_existential_node(p) for p in protein_names]
        ex_protein_idxs = th.tensor([self.node_to_id[p] for p in existential_protein_names], dtype=th.long, device=self.device)

        self._existential_protein_idxs = ex_protein_idxs
        return self._existential_protein_idxs


    
    @property
    def training_interactions_path(self):
        filename = f"train.tsv"
        path = os.path.join(self.root, filename)
        return path

    def get_existential_node(self, node):
        rel = "http://interacts"
        return f"{rel} some {node}"
        return f"DOMAIN_{rel}_under_{rel} some {node}"
    
    def create_subsumption_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep="\t", header=None)
        tuples.columns = ["head", "tail"]

        rel_name = "http://interacts"

        heads = ["http://"+ h for h in tuples["head"].values]
        tails = ["http://"+ t for t in tuples["tail"].values]

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
        assert num_testing_heads == num_testing_tails, "Heads and tails should be the same"
        
        subsumption_relation = subsumption_rel_name[self.graph_type]
        
        self.eval_relations = {subsumption_relation: 0} # this variable is defined here for the first time and it is used later in compute_ranking_metrics function

        num_eval_relations = len(self.eval_relations)
        filtering_labels = np.ones((num_eval_relations, num_testing_heads, num_testing_tails), dtype=np.int16)

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
                    rel_name = "http://arrow"
                    rel_idx = self.eval_relations[rel_name]
                    tail_graph_id = tail_idxs[i]
                    tail_ont_id = th.where(self.existential_protein_idxs == tail_graph_id)[0]
                    filtering_labels[rel_idx, head_ont_id, tail_ont_id] = 10000
                    
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
        ranks, filtered_ranks = dict(), dict()

        subsumption_relation = subsumption_rel_name[self.graph_type]
        self.eval_relations = {subsumption_relation: 0}

        if mode == "test":
            mrr, filtered_mrr = 0, 0
            hits_at_1, fhits_at_1 = 0, 0
            hits_at_3, fhits_at_3 = 0, 0
            hits_at_10, fhits_at_10 = 0, 0
            hits_at_100, fhits_at_100 = 0, 0
        
        tuples_path = self.test_tuples_path if mode == "test" else self.validation_tuples_path
        testing_dataloader = self.create_subsumption_dataloader(tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics..."):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs)
                
                for i, graph_head in enumerate(head_idxs):
                    head = th.where(self.protein_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.existential_protein_idxs == graph_tail)[0]
            

                    rel = rel_idxs[i]
                    eval_rel = self.eval_relations[self.id_to_relation[rel.item()]]
                        
                    preds = predictions[i]

                    orderings = th.argsort(preds, descending=True)
                    try:
                        rank = th.where(orderings == tail)[0].item()
                    except Exception as e:
                        print(tail)
                        print(self.id_to_node[graph_tail.item()])
                        raise e
                    mean_rank += rank
                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if mode == "test":
                        filt_labels = filtering_labels[eval_rel, head, :]
                        filt_labels[tail] = 1
                        filtered_preds = preds.cpu().numpy() * filt_labels
                        filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_orderings = th.argsort(filtered_preds, descending=True) 
                        filtered_rank = th.where(filtered_orderings == tail)[0].item()
                        filtered_mean_rank += filtered_rank
                    
                        mrr += 1/(rank+1)
                        filtered_mrr += 1/(filtered_rank+1)
                    
                        if rank == 0:
                            hits_at_1 += 1
                        if rank < 3:
                            hits_at_3 += 1
                        if rank < 10:
                            hits_at_10 += 1
                        if rank < 100:
                            hits_at_100 += 1


                        if filtered_rank == 0:
                            fhits_at_1 += 1
                        if filtered_rank < 3:
                            fhits_at_3 += 1
                        if filtered_rank < 10:
                            fhits_at_10 += 1
                        if filtered_rank < 100:
                            fhits_at_100 += 1

                        if filtered_rank not in filtered_ranks:
                            filtered_ranks[filtered_rank] = 0
                        filtered_ranks[filtered_rank] += 1

            mean_rank /= testing_dataloader.dataset_len

            if mode == "test":
                mrr /= testing_dataloader.dataset_len
                hits_at_1 /= testing_dataloader.dataset_len
                hits_at_3 /= testing_dataloader.dataset_len
                hits_at_10 /= testing_dataloader.dataset_len
                hits_at_100 /= testing_dataloader.dataset_len
                auc = self.compute_rank_roc(ranks)

                filtered_mean_rank /= testing_dataloader.dataset_len
                filtered_mrr /= testing_dataloader.dataset_len
                fhits_at_1 /= testing_dataloader.dataset_len
                fhits_at_3 /= testing_dataloader.dataset_len
                fhits_at_10 /= testing_dataloader.dataset_len
                fhits_at_100 /= testing_dataloader.dataset_len
                fauc = self.compute_rank_roc(filtered_ranks)

                raw_metrics = (mean_rank, mrr, hits_at_1, hits_at_3, hits_at_10, hits_at_100, auc)
                filtered_metrics = (filtered_mean_rank, filtered_mrr, fhits_at_1, fhits_at_3, fhits_at_10, fhits_at_100, fauc)

        if mode == "test":
            return raw_metrics, filtered_metrics
        else:
            return mean_rank
        
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
        logits = self.model.predict((head_idxs, rel_idxs, tail_idxs))
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
        logging.info("Testing unsatisfiability...")
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
