from src.models.cat import CatModel
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT
import os
import pandas as pd
from src.utils import FastTensorDataLoader

subsumption_rel_name = {
    "cat": "http://arrow",
    "cat1": "http://arrow",
    "cat2": "http://arrow"
}

class CatCompletion(CatModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ancestors_path = None
        if not self.test_completion:
            raise ValueError("test_completion must be True")
        
    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        
        graph_path = os.path.join(self.root, f"go.train.cat.edgelist")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        print("Graph path", graph_path)
        return self._graph_path

    @property
    def ancestors_path(self):
        if self._ancestors_path is not None:
            return self._ancestors_path
        filename = f"inferred_ancestors.txt"
        path = os.path.join(self.root, filename)
        assert os.path.exists(path)
        self._ancestors_path = path
        return self._ancestors_path

    
    def create_subsumption_dataloader(self, tuples_path, batch_size):
        data = []
        with open(tuples_path) as f:
            for line in f.readlines():
                line = line.rstrip("\n").split(",")
                head = line[0]
                tail = line[1:]
                for t in tail:
                    data.append((head, t))
        tuples = pd.DataFrame(data, columns=["head", "tail"])
            
        num_cols = tuples.shape[1]

                
        if num_cols == 2:
            tuples.columns = ["head", "tail"]
        elif num_cols == 3:
            tuples.columns = ["head", "relation", "tail"]
        else:
            raise ValueError(f"Invalid number of columns in {tuples_path}")

        tuples = tuples.drop_duplicates()
        tuples["head"] = tuples["head"].apply(lambda x: 'owl:Nothing' if x == "http://www.w3.org/2002/07/owl#Nothing" else x)
        tuples["head"] = tuples["head"].apply(lambda x: 'owl:Thing' if x == "http://www.w3.org/2002/07/owl#Thing" else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: 'owl:Nothing' if x == "http://www.w3.org/2002/07/owl#Nothing" else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: 'owl:Thing' if x == "http://www.w3.org/2002/07/owl#Thing" else x)
        
        heads = [self.node_to_id[h] for h in tuples["head"]]
        tails = [self.node_to_id[t] for t in tuples["tail"]]

        

        heads = th.tensor(heads, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)
        
        if num_cols == 2:
            rel_idx = self.relation_to_id[subsumption_rel_name[self.graph_type]]
            rels = rel_idx * th.ones_like(heads)
        else:
            
            if (self.graph_type in ["rdf", "cat", "cat1", "cat2"]) and self.test_unsatisfiability:
                rels = [self.node_to_id[r] for r in tuples["relation"]]
            else:
                rels = [self.relation_to_id[r] for r in tuples["relation"]]
            rels = th.tensor(rels, dtype=th.long)
        
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

        
    def get_filtering_labels(self):
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.ontology_classes_idxs)
        num_testing_tails = len(self.ontology_classes_idxs)
        
        subsumption_relation = subsumption_rel_name[self.graph_type]
        
        self.eval_relations = {subsumption_relation: 0} # this variable is defined here for the first time and it is used later in compute_ranking_metrics function

        num_eval_relations = len(self.eval_relations)
        filtering_labels = np.ones((num_eval_relations, num_testing_heads, num_testing_tails), dtype=np.int16)

        logging.debug(f"filtering_labels.shape: {filtering_labels.shape}")
                
        all_head_idxs = self.ontology_classes_idxs.to(self.device)
        all_tail_idxs = self.ontology_classes_idxs.to(self.device)
        eval_rel_idx = None

        testing_dataloader = self.create_subsumption_dataloader(self.ancestors_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    head_ont_id = th.where(self.ontology_classes_idxs == head_graph_id)[0]
                    rel = rel_idxs[i]

                    rel_name = self.id_to_relation[rel.item()]

                    rel_idx = self.eval_relations[rel_name]
                    tail_graph_id = tail_idxs[i]
                    tail_ont_id = th.where(self.ontology_classes_idxs == tail_graph_id)[0]
                    filtering_labels[rel_idx, head_ont_id, tail_ont_id] = 10000
                    
        return filtering_labels


    def compute_ranking_metrics(self, filtering_labels):
        print(f"Loading best model from {self.model_path}")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        mean_rank, filtered_mean_rank = 0, 0
        mrr, filtered_mrr = 0, 0
        hits_at_1, fhits_at_1 = 0, 0
        hits_at_3, fhits_at_3 = 0, 0
        hits_at_10, fhits_at_10 = 0, 0
        hits_at_100, fhits_at_100 = 0, 0
        ranks, filtered_ranks = dict(), dict()

        testing_dataloader = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics..."):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs)
                
                for i, graph_head in enumerate(head_idxs):
                    head = th.where(self.ontology_classes_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.ontology_classes_idxs == graph_tail)[0]

                    rel = rel_idxs[i]
                                                                
                    eval_rel = self.eval_relations[self.id_to_relation[rel.item()]]
                        
                    preds = predictions[i].cpu().numpy()

                    filt_labels = filtering_labels[eval_rel, head, :]
                    filt_labels[tail] = 1
                    filtered_preds = preds * filt_labels
                                                            

                    preds = th.from_numpy(preds).to(self.device)
                    filtered_preds = th.from_numpy(filtered_preds).to(self.device)

                    orderings = th.argsort(preds, descending=True)
                    filtered_orderings = th.argsort(filtered_preds, descending=True) 

                    rank = th.where(orderings == tail)[0].item()
                    filtered_rank = th.where(filtered_orderings == tail)[0].item()
                    
                    mean_rank += rank
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

                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

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
        return raw_metrics, filtered_metrics
                                                                                     
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
        logits = self.model.predict((head_idxs, rel_idxs, tail_idxs))
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
        


    
    def test(self):
        logging.info("Testing ontology completion...")
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
