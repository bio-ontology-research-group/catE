import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")

import click as ck
import os
from src.el_sem_membership import ELModel

from mowl.utils.random import seed_everything
import logging
import wandb


@ck.command()
@ck.option('--use_case', '-case', required=True, type=ck.Choice(["ore1"]))
@ck.option('--model_type', '-model', required=True, type=ck.Choice(["elem", "elbe", "box2el"]))
@ck.option('--num_models', '-nm', default=1)
@ck.option('--emb_dim', '-dim', required=True, type=int, default=256)
@ck.option('--batch_size', '-bs', required=True, type=int, default=128)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--margin', '-m', required=True, type=float, default=0.5)
@ck.option('--num_negs', '-negs', required=True, type=int, default=4)
@ck.option('--test_batch_size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=10000)
@ck.option('--device', '-d', default='cuda')
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--aggregator", '-agg', type=ck.Choice(['mean', 'min', 'max', 'median']), default='mean')
@ck.option("--no_sweep", '-ns', is_flag=True)
@ck.option("--description", '-desc', type=str, default='default')
def main(use_case, model_type, num_models, emb_dim, batch_size,
         lr, margin, num_negs, test_batch_size, epochs, device, seed,
         only_train, only_test, aggregator, description, no_sweep):

    wandb_logger = wandb.init(project='cate2', group=f'{use_case}_{model_type}', name=f'{description}')
    
    if no_sweep:
        wandb.log({'emb_dim': emb_dim,
                   'batch_size': batch_size,
                   'lr': lr,
                   'margin': margin
                   })
    else:    
        emb_dim = wandb.config.emb_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        margin = wandb.config.margin

    root = f"{use_case}/data"
            
    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\tmodel_type: ", model_type)
    print("\troot: ", root)
    print("\tnum_models: ", num_models)
    print("\temb_dim: ", emb_dim)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tmargin: ", margin)
    print("\tnum_negs: ", num_negs)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tepochs: ", epochs)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\taggregator: ", aggregator)
    seed_everything(seed)


    model_filepath = os.path.join(models_dir, f"{use_case}_{model_type}_{emb_dim}_{batch_size}_{lr}_{margin}.pt")
    model = ELModel(use_case,
                    model_type,
                    root,
                    num_models,
                    emb_dim,
                    batch_size,
                    lr,
                    margin,
                    test_batch_size,
                    epochs,
                    model_filepath,
                    device,
                    aggregator=aggregator)
        
        
    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        raw_metrics, filtered_metrics = model.test()
        wandb_logger.log(raw_metrics)
        wandb_logger.log(filtered_metrics)
                


        
if __name__ == "__main__":
    main()




 
