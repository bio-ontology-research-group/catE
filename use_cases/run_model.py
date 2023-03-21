import sys
sys.path.append('../')
sys.path.append("../../")

import mowl
mowl.init_jvm("10g")
import click as ck
import os
from src.models.baseline import Baseline
from src.models.baseline_unsat import BaselineUnsat
from src.models.cat_unsat import CatUnsat
from src.utils import seed_everything
import gc
import torch as th

@ck.command()
@ck.option('--use-case', '-case', required=True, type=ck.Choice(["pizza", "dideo", "fobi"]))
@ck.option('--graph-type', '-g', required=True, type=ck.Choice(['rdf', "owl2vec", 'onto2graph', 'cat', 'cat1', 'cat2']))
@ck.option('--kge-model', '-kge', required=True, type=ck.Choice(['transe', 'transr', 'ordere', 'transd']))
@ck.option('--root', '-r', required=True, type=ck.Path(exists=True))
@ck.option('--emb-dim', '-dim', required=True, type=int, default=256)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--weight-decay', '-wd', required=True, type=float, default = 0.0)
@ck.option('--batch-size', '-bs', required=True, type=int, default=4096*8)
@ck.option('--lr', '-lr', required=True, type=float, default=0.001)
@ck.option('--num-negs', '-negs', required=True, type=int, default=4)
@ck.option('--test-batch-size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=300)
@ck.option('--test-unsatisfiability', '-tu', is_flag=True)
@ck.option('--test_file', '-tf', required=True, type=ck.Path(exists=True))
@ck.option('--device', '-d', required=True, type=ck.Choice(['cpu', 'cuda']))
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option('--result-filename', '-rf', required=True)
def main(use_case, graph_type, kge_model, root, emb_dim, margin,
         weight_decay, batch_size, lr, num_negs, test_batch_size,
         epochs, test_unsatisfiability, test_file, device, seed,
         only_train, only_test, result_filename):

    if not result_filename.endswith('.csv'):
        raise ValueError("For convenience, please specify a csv file as result_filename")

    if root.endswith('/'):
        root = root[:-1]

    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\tgraph_type: ", graph_type)
    print("\tkge_model: ", kge_model)
    print("\troot: ", root)
    print("\temb_dim: ", emb_dim)
    print("\tmargin: ", margin)
    print("\tweight_decay: ", weight_decay)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tnum_negs: ", num_negs)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tepochs: ", epochs)
    print("\ttest_unsatisfiability: ", test_unsatisfiability)
    print("\ttest_file: ", test_file)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\tresult_filename: ", result_filename)
    seed_everything(seed)

    if test_unsatisfiability:
        if graph_type in ["cat", "cat1", "cat2"]:
            Model = CatUnsat
        else:
            Model = BaselineUnsat
    else:
        Model = Baseline
        
    model = Model(use_case,
                  graph_type,
                  kge_model,
                  root,
                  emb_dim,
                  margin,
                  weight_decay,
                  batch_size,
                  lr,
                  num_negs,
                  test_batch_size,
                  epochs,
                  test_file,
                  device,
                  seed,
                  10, #tolerance,
                  test_unsatisfiability,
                  )

    if not only_test:
        model.train()

    if not only_train:
        assert os.path.exists(test_file)
        params = (emb_dim, margin, weight_decay, batch_size, lr, num_negs)

        if test_unsatisfiability:
            print("Start testing unsatisfiability")
            raw_metrics, filtered_metrics = model.test()
            save_results(params, raw_metrics, filtered_metrics, result_filename)

def save_results(params, raw_metrics, filtered_metrics, result_dir):
    emb_dim, margin, weight_decay, batch_size, lr, num_negs = params
    mr, mrr, h1, h3, h10, h100, auc = raw_metrics
    mr_f, mrr_f, h1_f, h3_f, h10_f, h100_f, auc_f = filtered_metrics
    with open(result_dir, 'a') as f:
        line = f"{emb_dim},{margin},{weight_decay},{batch_size},{mr},{mrr},{h1},{h3},{h10},{h100},{auc},{mr_f},{mrr_f},{h1_f},{h3_f},{h10_f},{h100_f},{auc_f}\n"
        f.write(line)
    print("Results saved to ", result_dir)
        
if __name__ == "__main__":
    main()




 
