import sys
sys.path.append('../')
sys.path.append("../../")
sys.path.append("../../../")

import mowl
mowl.init_jvm("10g")
import click as ck
import os
from src.cat_completion_el import CatCompletion
from mowl.utils.random import seed_everything
import gc
import torch as th

import wandb

@ck.command()
@ck.option('--use_case', '-case', required=True, type=ck.Choice(["go", "foodon"]))
@ck.option('--emb_dim', '-dim', required=True, type=int, default=256)
@ck.option('--batch_size', '-bs', required=True, type=int, default=4096*8)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num_negs', '-negs', required=True, type=int, default=4)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--loss_type', '-loss', required=True, type=ck.Choice(["bpr", "normal"]))
@ck.option('--test_batch_size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=10000)
@ck.option('--device', '-d', default="cuda")
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--not_sweep", '-ns', is_flag=True)
@ck.option("--description", '-desc', default="default")
def main(use_case, emb_dim, batch_size, lr, num_negs, margin,
         loss_type, test_batch_size, epochs, device, seed, only_train,
         only_test, not_sweep, description):


    wandb_logger = wandb.init(project="cate2", name=description, group=f"cat_sat_{use_case}")

    if not_sweep:
        wandb_logger.log({"emb_dim": emb_dim,
                          "batch_size": batch_size,
                          "lr": lr,
                          "num_negs": num_negs,
                          "margin": margin,
                          "loss_type": loss_type,
                          })
    else:
        emb_dim = wandb.config.emb_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_negs = wandb.config.num_negs
        margin = wandb.config.margin
        loss_type = wandb.config.loss_type

    
    root = f"{use_case}/data"
    test_file = os.path.join(root, "test_cleaned.csv")
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
    print("\tloss_type: ", loss_type)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tepochs: ", epochs)
    print("\ttest_file: ", test_file)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    seed_everything(seed)

    validation_file = os.path.join(root, "valid_cleaned.csv")
    model = CatCompletion(use_case,
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
                          test_completion=True,
                  )

    if not only_test:
        model.train(wandb_logger)

    if not only_train:
        
        raw_metrics, filtered_metrics = model.test()
        wandb_logger.log(raw_metrics)
        wandb_logger.log(filtered_metrics)
        
            
            
if __name__ == "__main__":
    main()




 
