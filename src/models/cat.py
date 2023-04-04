from src.models.base_model import Model, subsumption_rel_name

import torch.optim as optim
import torch as th
import torch.nn as nn
from tqdm import trange, tqdm
import logging


class CatModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
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
        best_mr = float("inf")
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)
        
        
        for epoch in trange(self.epochs, desc=f"Training..."):
                            
            logging.info(f"Epoch: {epoch+1}")
            self.model.train()

            graph_loss = 0
            for head, rel, tail in tqdm(graph_dataloader, desc="Processing batches"):
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)
                
                data = (head, rel, tail)
                pos_logits = self.model.forward(data)

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data)
                neg_logits /= self.num_negs

                if self.kge_model == "ordere":
                    #batch_loss = (-pos_logits + th.relu(self.margin + neg_logits)).mean()
                    batch_loss = -criterion_bpr(pos_logits - neg_logits - self.margin).mean()
                else:
                    #batch_loss = -criterion_bpr(-pos_logits + neg_logits + self.margin).mean()
                    #batch_loss = -criterion_bpr(-pos_logits + neg_logits + self.margin).mean()
                    batch_loss = -criterion_bpr(pos_logits - neg_logits - self.margin).mean()
                    #batch_loss = th.relu(pos_logits - neg_logits + self.margin).mean()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(graph_dataloader)

            if self.able_to_validate:
                valid_mean_rank = self.compute_ranking_metrics(mode="validate")
                if valid_mean_rank < best_mr:
                    best_mr = valid_mean_rank
                    th.save(self.model.state_dict(), self.model_path)
                    tolerance = self.initial_tolerance+1
                    print("Model saved")
                    print(f"Training loss: {graph_loss:.6f}\tValidation mean rank: {valid_mean_rank:.6f}")

            else:   
                if best_loss > graph_loss:
                    best_loss = graph_loss
                    th.save(self.model.state_dict(), self.model_path)
                    tolerance = self.initial_tolerance+1
                    print("Model saved")
                    print(f"Training loss: {graph_loss:.6f}\n")
                
            tolerance -= 1
            if tolerance == 0:
                print("Early stopping")
                break

                


    
