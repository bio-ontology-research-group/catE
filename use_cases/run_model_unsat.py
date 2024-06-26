import sys
sys.path.append('../')
sys.path.append("../../")
sys.path.append("../../../")

import mowl
mowl.init_jvm("10g")
import click as ck
import os
from src.models.baseline import Baseline
from src.models.baseline_unsat import BaselineUnsat
from src.models.baseline_completion import BaselineCompletion
from src.models.cat_ppi import CatPPI
from src.models.cat_unsat import CatUnsat
from src.models.cat_deductive import CatDeductive
from src.models.cat_completion import CatCompletion
from src.utils import seed_everything
import gc
import torch as th

@ck.command()
@ck.option('--use-case', '-case', required=True, type=ck.Choice(["pizza", "nro", "kisao", "dideo", "fobi", "go_comp", "foodon_comp", "go_ded", "ppi"]))
@ck.option('--graph-type', '-g', required=True, type=ck.Choice(['rdf', "owl2vec", 'onto2graph', 'cat', 'cat1', 'cat2', 'catnit']))
@ck.option('--kge-model', '-kge', required=True, type=ck.Choice(['transe', 'distmult', 'convkb', 'ordere']))
@ck.option('--root', '-r', required=True, type=ck.Path(exists=True))
@ck.option('--emb-dim', '-dim', required=True, type=int, default=256)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--weight-decay', '-wd', required=True, type=float, default = 0.0)
@ck.option('--batch-size', '-bs', required=True, type=int, default=4096*8)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num-negs', '-negs', required=True, type=int, default=4)
@ck.option('--test-batch-size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=300)
@ck.option('--test-unsatisfiability', '-tu', is_flag=True)
@ck.option('--test-deductive-inference', '-td', is_flag=True)
@ck.option('--test-ontology-completion', '-tc', is_flag=True)
@ck.option('--test-named-classes', '-tn', is_flag=True)
@ck.option('--reduced-subsumption', '-rs', is_flag=True)
@ck.option('--test-existentials', '-te', is_flag=True)
@ck.option('--test-both-quantifiers', '-tbq', is_flag=True)
@ck.option('--test-ppi', '-tppi', is_flag=True)
@ck.option('--validation-file', '-vf', type=ck.Path(exists=True), default=None)
@ck.option('--test-file', '-tf', required=True, type=ck.Path(exists=True))
@ck.option('--device', '-d', required=True, type=ck.Choice(['cpu', 'cuda']))
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option('--result-filename', '-rf', required=True)
def main(use_case, graph_type, kge_model, root, emb_dim, margin, weight_decay, batch_size, lr, num_negs,
         test_batch_size, epochs, test_unsatisfiability, test_deductive_inference, test_ontology_completion,
         test_named_classes, reduced_subsumption, test_existentials, test_both_quantifiers, test_ppi, validation_file, test_file, device,
         seed, only_train, only_test, result_filename):

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
    print("\ttest_deductive_inference: ", test_deductive_inference)
    print("\ttest_ontology_completion: ", test_ontology_completion)
    print("\ttest_named_classes: ", test_named_classes)
    print("\treduced_subsumption: ", reduced_subsumption)
    print("\ttest_existentials: ", test_existentials)
    print("\ttest_both_quantifiers: ", test_both_quantifiers)
    print("\ttest_ppi: ", test_ppi)
    print("\tvalidation_file: ", validation_file)
    print("\ttest_file: ", test_file)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\tresult_filename: ", result_filename)
    seed_everything(seed)

    if test_unsatisfiability:
        if "cat" in graph_type:
            Model = CatUnsat
        else:
            Model = BaselineUnsat
    elif test_deductive_inference:
        if "cat" in graph_type:
            Model = CatDeductive
    elif test_ontology_completion:
        if "cat" in graph_type:
            Model = CatCompletion
        else:
            Model = BaselineCompletion
    elif test_ppi:
        Model = CatPPI
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
                  validation_file,
                  test_file,
                  device,
                  seed,
                  10, #tolerance,
                  test_unsatisfiability,
                  test_deductive_inference,
                  test_ontology_completion,
                  test_named_classes,
                  reduced_subsumption,
                  test_existentials,
                  test_ppi
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
        elif test_deductive_inference:
            print("Start testing deductive inference")
            raw_metrics, filtered_metrics = model.test(test_named_classes, test_existentials, test_both_quantifiers)
            save_results(params, raw_metrics, filtered_metrics, result_filename)
        elif test_ontology_completion:
            print("Start testing ontology completion")
            raw_metrics, filtered_metrics = model.test()
            save_results(params, raw_metrics, filtered_metrics, result_filename)
        elif test_ppi:
            print("Start testing PPI")
            raw_metrics, filtered_metrics = model.test()
            save_results(params, raw_metrics, filtered_metrics, result_filename)
            
            
def save_results(params, raw_metrics, filtered_metrics, result_dir):
    emb_dim, margin, weight_decay, batch_size, lr, num_negs = params
    mr, mrr, h1, h3, h10, h50, h100, auc = raw_metrics
    mr_f, mrr_f, h1_f, h3_f, h10_f, h50_f, h100_f, auc_f = filtered_metrics
    with open(result_dir, 'a') as f:
        line = f"{emb_dim},{margin},{weight_decay},{batch_size},{mr},{mrr},{h1},{h3},{h10},{h50},{h100},{auc},{mr_f},{mrr_f},{h1_f},{h3_f},{h10_f},{h50_f},{h100_f},{auc_f}\n"
        f.write(line)
    print("Results saved to ", result_dir)
        
if __name__ == "__main__":
    main()




 
