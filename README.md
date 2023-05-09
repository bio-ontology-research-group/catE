# CatE: Embedding $\mathcal{ALC}$ ontologies using category-theoretical semantics.

## Abstract

Machine learning with Semantic Web ontologies follows several
strategies, one of which involves projecting ontologies into graph
structures and applying graph embeddings or graph-based machine
learning methods to the resulting graphs. Several methods have been
developed that project ontology axioms into graphs. However, these
methods are limited in the type of axioms they can project (totality),
whether they are invertible (injectivity), and how they exploit
semantic information. These limitations restrict the kind of tasks to
which they can be applied. Category-theoretical semantics of logic
languages formalizes interpretations using categories instead of sets,
and categories have a graph-like structure.  We developed CatE, which
uses the category-theoretical formulation of the semantics of the
Description Logic ALC to generate a graph representation for ontology
axioms. The CatE projection is total and injective, and therefore
overcomes limitations of other graph-based ontology embedding methods
which are generally not invertible. We apply CatE to a number of
different tasks, including deductive and inductive reasoning, and we
demonstrate that CatE improves over state of the art ontology
embedding methods. Furthermore, we show that CatE can also outperform
model-theoretic ontology embedding methods in machine learning tasks
in the biomedical domain.

## Repository Overview

```

├── README.md
├── run
├── src
│   ├── cat
│   ├── data
│   ├── models
│   ├── projectors
│   └── visual
├── tests
│   ├── example3
│   └── example4
└── use_cases
    ├── experiments
    │   ├── dideo
    │   ├── fobi
    │   ├── foodon_completion
    │   ├── go_completion
    │   ├── go_deductive
    │   ├── kisao
    │   ├── nro
    │   ├── pizza
    │   └── ppi
    ├── fobi
    │   ├── data
    │   └── models
    ├── foodon_completion
    │   ├── data
    │   └── models
    ├── go
    │   ├── data
    │   └── models
    ├── go_completion
    │   ├── data
    │   └── models
    ├── go_deductive
    │   ├── data
    │   └── models
    ├── nro
    │   ├── data
    │   └── models
    └── ppi
        ├── data
        └── models


```


### Dependencies

* Python 3.8
* Anaconda
* Git LFS

### Set up environment



```
git lfs install #due to some large data files
git clone --recursive https://github.com/bio-ontology-research-group/catE.git
cd catE
conda env create -f environment.yml
conda activate cate
```

## Getting the data

The data is located in the `use_cases` directory for each task. For example, You will find the files for FoodOn completion task in `use_cases/foodon_completion/data.tar.gz`. To uncompress the data just run:

```
cd use_cases/foodon_completion
tar -xzvf data.tar.gz
```

## Running the model


To run the script, use the ``run_model.py`` script. The parameters are the following:

| Parameter                | Command line | Options                                                                      |
|--------------------------|--------------|------------------------------------------------------------------------------|
| case                     | -case        | nro, fobi, go_comp, foodon_comp, go_ded, ppi                                 |
| graph                    | -g           | owl2vec, onto2graph, rdf, cat, cat1, cat2                                    |
| kge model                | -kge         | transe, ordere                                                               |
| root directory           | -r           | [case/data                                                                   |
| embedding dimension      | -dim         | 64, 128, 256, ...                                                            |
| margin                   | -margin      | 0.0, 0.02, 0.04, ...                                                         |
| weight decay             | -wd          | 0.0000, 0.0001, 0.0005                                                       |
| batch size               | -bs          | 4096, 8192, 16834                                                            |
| learning rate            | -lr          | 0.1, 0.01, 0.001                                                             |
| testing batch size       | -tbs         | 8, 16                                                                        |
| number of negatives      | -negs        | 1,2,3,...                                                                    |
| epochs                   | -e           | 1000, 2000, ...                                                              |
| device                   | -d           | cpu, cuda                                                                    |
| validation file          | -vf          | name\_of\_validation\_file.csv                                               |
| testing file             | -tf          | name\_of\_testing\_file.csv                                                  |
| results file             | -rf          | name\_of\_results\_csv\_file.csv                                             |
| reduced subsumption      | -rs          | Train on ontology with some axioms $C \sqsubseteq D$ removed                 |
| test unsatisfiability    | -tu          | Flag to test $C \sqsubseteq \bot$ axioms                                     |
| test deductive inference | -td          | Flag to test $C \sqsubseteq D$ axioms in the deductive closure               |
| test ontology completion | -tc          | Flag to test $C \sqsubseteq D$ axioms in that are plausible but not entailed |
| test ppi                 | -tppi        | Flag to test ppi prediction axioms                                           |
| only train               | -otr         | Flag to only perform training                                                |
| only test                | -ot          | Flag to only perform testing                                                 |
| seed                     | -s           | 42                                                                           |

For example, to train the ontology completion task for FoodOn you can run
```
python run_model.py -case foodon_comp -g cat -kge ordere -r foodon_comp/data -dim 64 -m 0.2 -wd 0.0001 -bs 4096 -lr 0.1 -tbs 8 -e 4000 -d cuda -rd results_foodon_comp.csv -tf foodon/data/test.csv 
```
The commands to run the experiments in the paper are located at `use_cases/experiments`
   
  
  
  
  
