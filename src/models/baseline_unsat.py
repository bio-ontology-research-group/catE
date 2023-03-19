from src.models.baseline import Baseline
from src.utils import subsumption_rel_name
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT

class BaselineUnsat(Baseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_filtering_labels(self):
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.ontology_classes_idxs)
        
        subsumption_relation = subsumption_rel_name[self.graph_type]
        
        self.eval_relations = {subsumption_relation: 0} # this variable is defined here for the first time and it is used later in compute_ranking_metrics function

        filtering_labels = np.ones((num_testing_heads, ), dtype=np.int16)

        logging.debug(f"filtering_labels.shape: {filtering_labels.shape}")
                
        all_head_idxs = self.ontology_classes_idxs.to(self.device)
        eval_rel_idx = self.relation_to_id[subsumption_relation]

        testing_dataloader = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    head_ont_id = th.where(self.ontology_classes_idxs == head_graph_id)[0]
                    rel = rel_idxs[i]
                    rel_name = self.id_to_relation[rel.item()]
                    
                    assert rel_name == subsumption_relation, f"{rel_name} != {subsumption_relation}"
                                                                
                    graph_rel_name = self.id_to_relation[rel.item()]
                    filtering_labels[head_ont_id] = 10000
        return filtering_labels


    def compute_ranking_metrics(self, filtering_labels):
        print(f"Loading best model from {self.model_path}")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        mean_rank, filtered_mean_rank = 0, 0
        mrr, filtered_mrr = 0, 0
        hits_at_1, fhits_at_1 = 0, 0
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
                    assert self.ontology_classes[tail] == BOT
                    preds = predictions[i].cpu().numpy()

                    filt_labels = filtering_labels.copy()
                    filt_labels[head] = 1
                    filtered_preds = preds * filt_labels
                                                            

                    preds = th.from_numpy(preds).to(self.device)
                    filtered_preds = th.from_numpy(filtered_preds).to(self.device)

                    orderings = th.argsort(preds, descending=False)
                    filtered_orderings = th.argsort(filtered_preds, descending=False)

                    rank = th.where(orderings == head)[0].item()
                                                                                            
                    filtered_rank = th.where(filtered_orderings == head)[0].item()
                    
                    mean_rank += rank
                    filtered_mean_rank += filtered_rank
                    
                    mrr += 1/(rank+1)
                    filtered_mrr += 1/(filtered_rank+1)
                    
                    if rank == 0:
                        hits_at_1 += 1
                    if rank < 10:
                        hits_at_10 += 1
                    if rank < 100:
                        hits_at_100 += 1

                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if filtered_rank == 0:
                        fhits_at_1 += 1
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
            hits_at_10 /= testing_dataloader.dataset_len
            hits_at_100 /= testing_dataloader.dataset_len
            auc = self.compute_rank_roc(ranks)

            filtered_mean_rank /= testing_dataloader.dataset_len
            filtered_mrr /= testing_dataloader.dataset_len
            fhits_at_1 /= testing_dataloader.dataset_len
            fhits_at_10 /= testing_dataloader.dataset_len
            fhits_at_100 /= testing_dataloader.dataset_len
            fauc = self.compute_rank_roc(filtered_ranks)

            raw_metrics = (mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100, auc)
            filtered_metrics = (filtered_mean_rank, filtered_mrr, fhits_at_1, fhits_at_10, fhits_at_100, fauc)
        return raw_metrics, filtered_metrics


    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
          logits = self.model.forward((head_idxs, rel_idxs, tail_idxs))
          logits = logits.reshape(-1, len(self.ontology_classes_idxs))
          return logits

    
    def predict(self, heads, rels, tails):

        aux = tails.to(self.device)
        num_tails = len(tails)
                
        tails = tails.to(self.device)
        tails = tails.repeat(len(self.ontology_classes_idxs),1).T
        assert (tails[0,:] == aux[0]).all(), f"{tails[0,:]}, {aux[0]}"
        tails = tails.reshape(-1)
        assert (aux[0] == tails[:num_tails]).all(), "tails are not the same"
        rels = rels.to(self.device)

        
        
        rels = rels.repeat(len(self.ontology_classes_idxs),1).T
        rels = rels.reshape(-1)
                                                
        eval_heads = self.ontology_classes_idxs.repeat(num_tails)
                
        logits = self.normal_forward(eval_heads, rels, tails)

        return logits
        


    
    def test(self):
        logging.info("Testing unsatisfiability...")
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
