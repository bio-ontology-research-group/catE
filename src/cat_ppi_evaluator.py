from src.cat import CatModel
import logging

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm
from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi import OWLAPIAdapter
import pandas as pd
import os
from scipy.stats import rankdata
from mowl.utils.data import FastTensorDataLoader
from mowl.datasets.builtin import PPIYeastDataset
from mowl.datasets import Dataset, OWLClasses, PathDataset
from itertools import cycle
from tqdm import trange, tqdm
from src.evaluators import PPIEvaluator, SubsumptionEvaluator

SUBSUMPTION_RELATION = "http://arrow"

class PPIYeastDatasetV2(PPIYeastDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deductive_closure_ontology = None
        
    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            root_dir = os.path.dirname(os.path.abspath(self.ontology_path))
            ontology_path = os.path.join(root_dir, "ontology_deductive_closure.owl")
            self._deductive_closure_ontology = PathDataset(ontology_path).ontology

        return self._deductive_closure_ontology

    @property
    def classes(self):
        """List of classes in the dataset. The classes are collected from training, validation and
        testing ontologies using the OWLAPI method ``ontology.getClassesInSignature()``.

        :rtype: OWLClasses
        """
        if self._classes is None:
            adapter = OWLAPIAdapter()
            top = adapter.create_class(TOP)
            bot = adapter.create_class(BOT)
            classes = set([top, bot])
            classes |= set(self._ontology.getClassesInSignature())

            # if self._validation:
                # classes |= set(self._validation.getClassesInSignature())
            # if self._testing:
                # classes |= set(self._testing.getClassesInSignature())

            classes = list(classes)
            self._classes = OWLClasses(classes)
        return self._classes
        
    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            train_classes = self.ontology.getClassesInSignature()
            train_classes = OWLClasses(train_classes).as_dict.keys()
            train_classes = set(train_classes)
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if not owl_name in train_classes:
                    continue
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes


class CatPPI(CatModel):

    def __init__(self, *args, evaluate_testing_set=True,
                 evaluate_with_deductive_closure=False,
                 filter_deductive_closure=False, **kwargs):
        self._loaded = False
        self._evaluator_loaded = False

        super().__init__(*args, **kwargs)

        self.evaluate_testing_set = evaluate_testing_set
        self.evaluate_with_deductive_closure = evaluate_with_deductive_closure
        self.filter_deductive_closure = filter_deductive_closure
        
        print(self.model)
        self._load()
        self._load_evaluator()
        self._protein_names = None
        self._protein_idxs = None
        self._existential_protein_idxs = None 

        print(f"Number of proteins: {len(self.protein_idxs)}")
        print(f"Number of existential proteins: {len(self.existential_protein_idxs)}")

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        graph_name = "ontology_extended.cat.edgelist" 
        # graph_name = "ontology.cat.edgelist"
        
        graph_path = os.path.join(self.root, f"{graph_name}")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        print("Graph path", graph_path)
        return self._graph_path


    def _load(self):

        # if self._loaded:
            # return
        
        ds = PPIYeastDatasetV2()
        classes = ds.classes.as_str
        
        ds = Dataset(ds.ontology)
        
        
        
        proteins = set()
        for owl_name, owl_cls in ds.classes.as_dict.items():
            if "http://4932" in owl_name:
                proteins.add(owl_cls)
        proteins = OWLClasses(proteins)
        proteins = proteins.as_str
        # print(proteins)
        # proteins.remove("http://4932.YCL020W")
        
        self._ontology_classes = classes
        self._protein_names = proteins
        self._loaded = True
        
    def _load_evaluator(self):
        if self._evaluator_loaded:
            return
        
        ds = PPIYeastDatasetV2()

        

        if self.evaluate_with_deductive_closure:
            self.evaluation_model = EvaluationModelSubsumption(self.model, ds,
                                                       self.node_to_id,
                                                       self.relation_to_id,
                                                       self.device)

            self.evaluator = SubsumptionEvaluator(ds, self.device, batch_size=128,
                                          evaluate_testing_set=self.evaluate_testing_set,
                                          evaluate_with_deductive_closure=self.evaluate_with_deductive_closure,
                                          filter_deductive_closure=self.filter_deductive_closure)

        else:
            self.evaluation_model = EvaluationModelPPI(self.model, ds,
                                                    self.node_to_id,
                                                    self.relation_to_id,
                                                    self.device)

            self.evaluator = PPIEvaluator(ds, self.device)

            
        self._evaluator_loaded = True
    
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

         
        
    def test(self):
        logging.info("Testing ppi...")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)
        metrics = self.evaluator.evaluate(self.evaluation_model, mode="test")
        print_as_md(metrics)
        
        return metrics
                        
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



    def train_ppi(self, wandb_logger):
        
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        min_lr = self.lr/10
        max_lr = self.lr
        print("Min lr: {}, Max lr: {}".format(min_lr, max_lr))
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        graph_dataloader = self.create_graph_train_dataloader(batch_size=16*self.batch_size)
        graph_dataloader = cycle(graph_dataloader)

        train_ppi_dataloader = self.create_subsumption_dataloader(self.training_interactions_path, batch_size=self.batch_size)

        

        tolerance = 0
        best_loss = float("inf")
        best_mr = float("inf")
        best_mrr = 0
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)

        for epoch in trange(self.epochs, desc=f"Training..."):
            # logging.info(f"Epoch: {epoch+1}")
            self.model.train()

            graph_loss = 0
            for ppi_head, ppi_rel, ppi_tail in train_ppi_dataloader:
                # ppi_head, ppi_rel, ppi_tail = next(train_ppi_dataloader)
                ppi_head = ppi_head.to(self.device)
                ppi_rel = ppi_rel.to(self.device)
                ppi_tail = ppi_tail.to(self.device)

                ppi_logits = self.model.forward(ppi_head, ppi_rel, ppi_tail)
                neg_tail_ids = th.randint(0, len(self.existential_protein_idxs), (len(ppi_head),), device=self.device)
                neg_tail = self.existential_protein_idxs[neg_tail_ids]
                ppi_neg_logits = self.model.forward(ppi_head, ppi_rel, neg_tail)

                if self.loss_type == "bpr":
                    ppi_loss = -criterion_bpr(ppi_logits - self.margin).mean()
                    ppi_loss += -criterion_bpr(self.margin - ppi_neg_logits).mean()
                elif self.loss_type == "normal":
                    ppi_loss = -ppi_logits.mean() + th.relu(self.margin + ppi_neg_logits).mean()

                batch_loss = ppi_loss

                head, rel, tail = next(graph_dataloader)
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
                    batch_loss = -criterion_bpr(pos_logits - self.margin).mean()
                elif self.loss_type == "normal":
                    batch_loss += -pos_logits.mean() + th.relu(self.margin + neg_logits).mean()

                # batch_loss += self.model.collect_regularization_term().mean()

                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()
                

            graph_loss /= len(train_ppi_dataloader)

            valid_every = 100

            if epoch % valid_every == 0:
                metrics = self.evaluator.evaluate(self.evaluation_model, mode="valid")
                valid_mean_rank = metrics["valid_mr"]
                valid_mrr = metrics["valid_mrr"]
            
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    th.save(self.model.state_dict(), self.model_path)
                    tolerance = self.initial_tolerance+1
                    print("Model saved")
                else:
                    tolerance -= 1


                    
                    
                print(f"Training loss: {graph_loss:.6f}\tValidation mean rank: {valid_mean_rank:.6f}\tValidation MRR: {valid_mrr:.6f}")
                wandb_logger.log({"epoch": epoch, "train_loss": graph_loss, "valid_mr": valid_mean_rank, "valid_mrr": valid_mrr})
            
                                    
            if tolerance == 0:
                print("Early stopping")
                break

                


    




    def train(self, wandb_logger):
        
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        min_lr = self.lr/10
        max_lr = self.lr
        print("Min lr: {}, Max lr: {}".format(min_lr, max_lr))
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        graph_dataloader = self.create_graph_train_dataloader()

        train_ppi_dataloader = self.create_subsumption_dataloader(self.training_interactions_path, batch_size=self.batch_size)

        train_ppi_dataloader = cycle(train_ppi_dataloader)

        
        
        tolerance = 0
        best_loss = float("inf")
        best_mr = float("inf")
        best_mrr = 0
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)

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

                # print(f"head.shape: {head.shape}. self.protein_idxs.shape: {self.protein_idxs.shape}")
                mask = th.isin(head, self.protein_idxs)
                pos_prot_logits = pos_logits[mask]
                head_prots = head[mask]
                neg_ids = th.randint(0, len(self.existential_protein_idxs), (len(head_prots),), device=self.device)
                neg_prots = self.existential_protein_idxs[neg_ids]
                neg_prot_logits = self.model.forward(head_prots, rel, neg_prots)
                

                
 
                if self.loss_type == "bpr":
                    batch_loss = -criterion_bpr(pos_logits - self.margin).mean()
                elif self.loss_type == "normal":
                    batch_loss = -pos_logits.mean() + th.relu(self.margin + neg_logits).mean()
                    batch_loss += -pos_prot_logits.mean() + th.relu(self.margin + neg_prot_logits).mean()
                #batch_loss += self.model.collect_regularization_term().mean()

                # ppi_head, ppi_rel, ppi_tail = next(train_ppi_dataloader)
                # ppi_head = ppi_head.to(self.device)
                # ppi_rel = ppi_rel.to(self.device)
                # ppi_tail = ppi_tail.to(self.device)

                # ppi_logits = self.model.forward(ppi_head, ppi_rel, ppi_tail)

                # ppi_neg_logits = 0
                # for i in range(self.num_negs):
                    # neg_tail = th.randint(0, len(self.existential_protein_idxs), (len(ppi_head),), device=self.device)
                    # ppi_neg_logits += self.model.forward(ppi_head, ppi_rel, neg_tail)

                # ppi_neg_logits /= self.num_negs
                    
                # if self.loss_type == "bpr":
                    # ppi_loss = -criterion_bpr(ppi_logits - self.margin).mean()
                    # ppi_loss += -criterion_bpr(self.margin - ppi_neg_logits).mean()
                # elif self.loss_type == "normal":
                    # ppi_loss = -ppi_logits.mean() + th.relu(self.margin + ppi_neg_logits).mean()

                # batch_loss += ppi_loss
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()
                

            graph_loss /= len(graph_dataloader)

            valid_every = 100
            if self.able_to_validate and epoch % valid_every == 0:
                metrics = self.evaluator.evaluate(self.evaluation_model, mode="valid")
                valid_mean_rank = metrics["valid_mr"]
                valid_mrr = metrics["valid_mrr"]
                
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    th.save(self.model.state_dict(), self.model_path)
                    tolerance = self.initial_tolerance+1
                    print("Model saved")
                else:
                    tolerance -= 1


                    
                    
                print(f"Training loss: {graph_loss:.6f}\tValidation mean rank: {valid_mean_rank:.6f}\tValidation MRR: {valid_mrr:.6f}")
                wandb_logger.log({"epoch": epoch, "train_loss": graph_loss, "valid_mr": valid_mean_rank, "valid_mrr": valid_mrr})
            
                                    
            if tolerance == 0:
                print("Early stopping")
                break

                


    



class EvaluationModelSubsumption(nn.Module):
    def __init__(self, kge_method, dataset, node_to_id, relation_to_id, device):
        super().__init__()
        
        self.device = device

        self.kge_method = kge_method

        evaluation_classes = dataset.classes.as_str
        evaluation_classes = [x for x in evaluation_classes if "GO_" in x]
                
        
        class_to_id = {ont: idx for idx, ont in enumerate(evaluation_classes)}
        self.id_to_class = {idx: ont for idx, ont in enumerate(evaluation_classes)}
                                 
        class_id_to_node_id = dict()

        for class_ in evaluation_classes:
            class_id_to_node_id[class_to_id[class_]] = node_to_id[class_]

        self.node_ids  = th.tensor(list(class_id_to_node_id.values()), dtype=th.long, device=self.device)
        
        relation_id = relation_to_id["http://arrow"]
        self.rel_embedding = th.tensor(relation_id).to(self.device)

                            
    def forward(self, data, *args, **kwargs):

        x = self.node_ids[data[:, 0]].unsqueeze(1)
        y = self.node_ids[data[:, 1]].unsqueeze(1)
                                
        r = self.rel_embedding.expand_as(x)
        
        triples = th.cat([x, r, y], dim=1)
        assert triples.shape[1] == 3
        scores = - self.kge_method.score_hrt(triples)
        # print(scores.min(), scores.max())
        return scores

class EvaluationModelPPI(nn.Module):
    def __init__(self, kge_method, dataset, node_to_id, relation_to_id, device):
        super().__init__()
        
        self.device = device

        self.kge_method = kge_method

        evaluation_heads, evaluation_tails = dataset.evaluation_classes
        evaluation_heads = evaluation_heads.as_str
        class_to_id = {ont: idx for idx, ont in enumerate(evaluation_heads)}
                                 
        ont_id_to_node_id = dict()

        for class_ in evaluation_heads:
            ont_id_to_node_id[class_to_id[class_]] = node_to_id[class_]

        ont_id_to_existential_id = dict()
        for class_ in evaluation_heads:
            ont_id_to_existential_id[class_to_id[class_]] = node_to_id[self.get_existential_node(class_)]
        
        self.node_ids  = th.tensor(list(ont_id_to_node_id.values()), dtype=th.long, device=self.device)
        self.existential_ids = th.tensor(list(ont_id_to_existential_id.values()), dtype=th.long, device=self.device)
        
        relation_id = relation_to_id["http://arrow"]
        self.rel_embedding = th.tensor(relation_id).to(self.device)

    def get_existential_node(self, node):
        rel = "http://interacts_with"
        return f"{rel} some {node}"
        #return f"DOMAIN_{rel}_under_{rel} some {node}"

    def forward(self, data, *args, **kwargs):

        x = self.node_ids[data[:, 0]].unsqueeze(1)
        y = self.existential_ids[data[:, 2]].unsqueeze(1)

                                
        r = self.rel_embedding.expand_as(x)
        
        triples = th.cat([x, r, y], dim=1)
        assert triples.shape[1] == 3
        scores = - self.kge_method.score_hrt(triples)
        # print(scores.min(), scores.max())
        return scores


    
def print_as_md(overall_metrics):
    
    metrics = ["test_mr", "test_mrr", "test_auc", "test_hits@1", "test_hits@3", "test_hits@10", "test_hits@50", "test_hits@100"]
    filt_metrics = [k.replace("_", "_f_") for k in metrics]

    string_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    string_filtered_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_filtered_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    
    string_metrics += "| Overall | "
    string_filtered_metrics += "| Overall | "
    for metric in metrics:
        if metric == "test_mr":
            string_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_metrics += f"{overall_metrics[metric]:.4f} | "
    for metric in filt_metrics:
        if metric == "test_f_mr":
            string_filtered_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_filtered_metrics += f"{overall_metrics[metric]:.4f} | "


    print(string_metrics)
    print("\n\n")
    print(string_filtered_metrics)
        

