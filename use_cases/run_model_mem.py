import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")

import click as ck
import os
from src.owl2vec_mem import OWL2Vec
from src.cate_mem import CatE
from src.el_sem_mem import ELModel

from src.utils import seed_everything
import logging
import wandb


@ck.command()
@ck.option('--use_case', '-case', required=True, type=ck.Choice(["ore1", "ore2", "ore3", "owl2bench1", "owl2bench2", "caligraph4", "caligraph5"]))
@ck.option('--model_type', '-model', required=True, type=ck.Choice(['owl2vec', "cat", "elem", "elbox", "box2el"]))
@ck.option('--kge_model', '-kge', type=ck.Choice(['transe', 'ordere', "transd", "distmult", "conve"]), default='transe')
@ck.option('--root', '-r', required=True, type=ck.Path(exists=True))
@ck.option('--num_models', '-nm', default=1)
@ck.option('--emb_dim', '-dim', required=True, type=int, default=256)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--weight_decay', '-wd', required=True, type=float, default = 0.0)
@ck.option('--batch_size', '-bs', required=True, type=int, default=128)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num_negs', '-negs', required=True, type=int, default=4)
@ck.option('--test_batch_size', '-tbs', required=True, type=int, default=32)
@ck.option('--alpha', '-a', required=True, type=float, default=0.5)
@ck.option('--epochs', '-e', required=True, type=int, default=10000)
@ck.option('--device', '-d', default='cuda')
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option('--only_sub', '-sub', is_flag=True)
@ck.option('--only_mem', '-mem', is_flag=True)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--aggregator", '-agg', type=ck.Choice(['mean', 'min', 'max', 'median']), default='mean')
@ck.option("--description", '-desc', type=str, default='default')
@ck.option("--no_sweep", '-ns', is_flag=True)
def main(use_case, model_type, kge_model, root, num_models, emb_dim,
         margin, weight_decay, batch_size, lr, num_negs,
         test_batch_size, alpha, epochs, device, seed, only_sub,
         only_mem, only_train, only_test, aggregator, description, no_sweep):

    wandb_logger = wandb.init(project='cate', group=f'{use_case}_{model_type}_{kge_model}', name=f'{description}')
    
    if no_sweep:
        wandb.log({'emb_dim': emb_dim, 'weight_decay': weight_decay,
                   'margin': margin, 'batch_size': batch_size,
                   'alpha': alpha})
    else:    
        emb_dim = wandb.config.emb_dim
        weight_decay = wandb.config.weight_decay
        margin = wandb.config.margin
        batch_size = wandb.config.batch_size
        alpha = wandb.config.alpha
    
    if root.endswith('/'):
        root = root[:-1]

    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\tmodel_type: ", model_type)
    print("\tkge_model: ", kge_model)
    print("\troot: ", root)
    print("\tnum_models: ", num_models)
    print("\temb_dim: ", emb_dim)
    print("\tmargin: ", margin)
    print("\tweight_decay: ", weight_decay)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tnum_negs: ", num_negs)
    print("\ttest_batch_size: ", test_batch_size)
    print("\talpha: ", alpha)
    print("\tepochs: ", epochs)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\taggregator: ", aggregator)
    print("\tonly_sub: ", only_sub)
    print("\tonly_mem: ", only_mem)
    seed_everything(seed)

    if model_type in ['owl2vec', 'cat']:
        if model_type == "owl2vec":
            graph_model = OWL2Vec
        elif model_type == 'cat':
            graph_model = CatE
        
        model = graph_model(use_case,
                        kge_model,
                        root,
                        emb_dim,
                        margin,
                        weight_decay,
                        batch_size,
                        lr,
                        num_negs,
                        test_batch_size,
                        alpha,
                        epochs,
                        device,
                        seed,
                        5, #tolerance,
                        )

    elif model_type in ["elem", "elbox", "box2el"]:
        wandb.log({'emb_dim': emb_dim, 'weight_decay': weight_decay,
                   'margin': margin, 'batch_size': batch_size,
                   'alpha': alpha})


        model_filepath = os.path.join(models_dir, f"{use_case}_{model_type}_sub_{only_sub}_mem_{only_mem}_{emb_dim}_{weight_decay}_{margin}_{batch_size}_{alpha}.pt")
        model = ELModel(use_case,
                        model_type,
                        root,
                        num_models,
                        emb_dim,
                        margin,
                        batch_size,
                        lr,
                        test_batch_size,
                        epochs,
                        model_filepath,
                        device,
                        only_sub,
                        only_mem,
                        aggregator=aggregator)
        
        
    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        subsumption_metrics, membership_metrics = model.test()
        log_results(subsumption_metrics, wandb_logger, prefix="sub")
        log_results(membership_metrics, wandb_logger, prefix="mem")
                                            
def log_results(metrics, logger, prefix):
    raw_metrics, filtered_metrics = metrics
    mr, mrr, median_rank, h1, h3, h10, h100, auc, perc90, below1000 = raw_metrics
    logger.log({f"{prefix}_mr": mr,
                f"{prefix}_mrr": mrr,
                f"{prefix}_median_rank": median_rank,
                f"{prefix}_h1": h1,
                f"{prefix}_h3": h3,
                f"{prefix}_h10": h10,
                f"{prefix}_h100": h100,
                f"{prefix}_auc": auc,
                f"{prefix}_perc90": perc90,
                f"{prefix}_below1000": below1000})
        
    mr_f, mrr_f, fmedian_rank, h1_f, h3_f, h10_f, h100_f, auc_f, fperc90, fbelow1000 = filtered_metrics
    logger.log({f"{prefix}_f_mr": mr_f,
                f"{prefix}_f_mrr": mrr_f,
                f"{prefix}_f_median_rank": fmedian_rank,
                f"{prefix}_f_h1": h1_f,
                f"{prefix}_f_h3": h3_f,
                f"{prefix}_f_h10": h10_f,
                f"{prefix}_f_h100": h100_f,
                f"{prefix}_f_auc": auc_f,
                f"{prefix}_f_perc90": fperc90,
                f"{prefix}_f_below1000": fbelow1000})

        
if __name__ == "__main__":
    main()




 
