import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")

import click as ck
import os
from src.cat_membership import CatMembership
from mowl.utils.random import seed_everything
import logging
import wandb


@ck.command()
@ck.option('--use_case', '-case', default="ore1", type=ck.Choice(["ore1"]))
@ck.option('--emb_dim', '-dim', required=True, type=int, default=256)
@ck.option('--batch_size', '-bs', required=True, type=int, default=128)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num_negs', '-negs', required=True, type=int, default=4)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--loss_type', "-loss", default="normal", type=ck.Choice(["normal", "bpr"]))
@ck.option('--test_batch_size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=10000)
@ck.option('--device', '-d', default='cuda')
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--description", '-desc', type=str, default='default')
@ck.option("--no_sweep", '-ns', is_flag=True)
def main(use_case, emb_dim, batch_size, lr, num_negs, margin,
         loss_type, test_batch_size, epochs, device, seed, only_train,
         only_test, description, no_sweep):

    wandb_logger = wandb.init(project='cate2', group=f'cat_mem_{use_case}', name=f'{description}')
    
    if no_sweep:
        wandb.log({'emb_dim': emb_dim,
                   'batch_size': batch_size,
                   'lr': lr,
                   'num_negs': num_negs,
                   'margin': margin,})
    else:    
        emb_dim = wandb.config.emb_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_negs = wandb.config.num_negs
        margin = wandb.config.margin
                

    root = f"{use_case}/data"
    if "ore1" in use_case:
        test_file = os.path.join(root, "ORE1_membership_test.edgelist")
            
    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\temb_dim: ", emb_dim)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tnum_negs: ", num_negs)
    print("\tmargin: ", margin)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tepochs: ", epochs)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    seed_everything(seed)

    if "ore1" in use_case:
        validation_file = os.path.join(root, "ORE1_membership_valid.edgelist")
    model = CatMembership(use_case,
                          False,
                          root,
                          emb_dim,
                          batch_size,
                          lr,
                          num_negs,
                          margin,
                          loss_type,
                          test_batch_size,
                          epochs,
                          validation_file,
                          test_file,
                          device,
                          seed,
                          10, #tolerance,
                          
                        )

                            
    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        raw_metrics, filtered_metrics = model.test()
        wandb.log(raw_metrics)
        wandb.log(filtered_metrics)


        
if __name__ == "__main__":
    main()




 
