from src.cat import CatModel
import logging

import numpy as np
import torch as th
from tqdm import tqdm
from mowl.owlapi.defaults import BOT
from src.utils import suffix_unsat

SUBSUMPTION_RELATION = "http://arrow"

class CatSat(CatModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)



    def get_filtering_labels(self):
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.ontology_classes_idxs)

        self.eval_relations = {subsumption_relation: 0} # this variable is defined here for the first time and it is used later in compute_ranking_metrics function

        filtering_labels = np.ones((num_testing_heads, ), dtype=np.int32)

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
                    filtering_labels[head_ont_id] = 10000 if DESCENDING else -10000

        filtering_labels[self.bot_idx] = 10000 if DESCENDING else -10000
        return filtering_labels


    def compute_ranking_metrics(self, filtering_labels=None, mode="test"):
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

        testing_dataloader = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics..."):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs)
                
                for i, graph_head in enumerate(head_idxs):
                    head = th.where(self.ontology_classes_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.ontology_classes_idxs == graph_tail)[0]
                    assert self.ontology_classes[tail] == "owl:Nothing"
                    preds = predictions[i].cpu().numpy()


                    if mode == "test":
                        filt_labels = filtering_labels.copy()
                        filt_labels[head] = 1
                        filtered_preds = preds * filt_labels
                                                            

                    preds = th.from_numpy(preds).to(self.device)
                    orderings = th.argsort(preds, descending=DESCENDING)
                    rank = th.where(orderings == head)[0].item() + 1

                    if mode == "test":
                        filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_orderings = th.argsort(filtered_preds, descending=DESCENDING)
                        filtered_rank = th.where(filtered_orderings == head)[0].item() + 1
                        filtered_mean_rank += filtered_rank
                        filtered_mrr += 1/(filtered_rank)


                    mean_rank += rank
                    mrr += 1/(rank)
                    

                    if mode == "test":

                        for k in hits_at_k.keys():
                            if rank < k:
                                hits_at_k[k] += 1
                            if filtered_rank < k:
                                fhits_at_k[k] += 1
                        
                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1

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
                for k in hits_at_k.keys():
                    raw_metrics[f"hits_{k}"] = hits_at_k[k]

                filtered_metrics = {"fmr": filtered_mean_rank, "fmrr": filtered_mrr, "fauc": fauc}
                for k in fhits_at_k.keys():
                    filtered_metrics[f"fhits_{k}"] = fhits_at_k[k]
                                
                return raw_metrics, filtered_metrics

            else:
                return mean_rank, mrr
        

    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
        data = th.vstack((head_idxs, rel_idxs, tail_idxs)).to(self.device)
        data = data.T
        
        logits = self.model.distance(data)
        logits = logits.reshape(-1, len(self.ontology_classes_idxs))
        return logits
                                            

    
    def predict(self, heads, rels, tails):

        aux = tails.to(self.device)
        num_tails = len(tails)
                
        tails = tails.to(self.device)
        tails = tails.repeat(len(self.ontology_classes_idxs),1).T
        assert (tails[0,:] == aux[0]).all(), f"{tails[0,:]}, {aux[0]}"
        tails = tails.reshape(-1)
        assert (aux[0] == tails[:len(self.ontology_classes)]).all(), "tails are not the same"
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
