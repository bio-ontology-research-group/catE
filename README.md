# CatE: Lattice preserving $\mathcal{ALC}$ ontology embeddings.

## DOI

https://doi.org/10.1007/978-3-031-71167-1_19



## Abstract
  
  Generating vector representations (embeddings) of OWL ontologies is
  a growing task due to its applications in predicting missing facts
  and knowledge-enhanced learning in fields such as
  bioinformatics. The underlying semantics of OWL ontologies is
  expressed using Description Logics (DLs). Initial approaches to
  generate embeddings relied on constructing a graph out of
  ontologies, neglecting the semantics of the logic therein. Recent
  semantic-preserving embedding methods often target lightweight DL
  languages like $\mathcal{EL}^{++}$, ignoring more expressive
  information in ontologies. Although some approaches aim to embed
  more descriptive DLs like $\mathcal{ALC}$, those methods require the
  existence of individuals, while many real-world ontologies are
  devoid of them. We propose an ontology embedding method for the
  $\mathcal{ALC}$ DL language that considers the lattice structure of
  concept descriptions. We use connections between DL and Category
  Theory to materialize the lattice structure and embed it using an
  order-preserving embedding method. We show that our method
  outperforms state-of-the-art methods in several knowledge base
  completion tasks.
      
### Dependencies

* Python >= 3.8
* mOWL

### Set up environment

```
cd catE
conda env create -f environment.yml
conda activate cate
```

## Getting the data

The data can be obtained from the following Zenodod repository: https://zenodo.org/records/13766937
After downloading, decompresss the file with the following command:

```
tar -xzvf use_cases.tar.gz
```

## Running the model
 
* ORE1

```
run_cat_membership.py --batch_size=32768 --emb_dim=200 --loss_type=normal --lr=0.0001 --margin=1 --num_negs=4 --use_case=ore1

```
 
* GO

```
python run_cat_completion.py --batch_size=32768 --emb_dim=200 --loss_type=normal --lr=1e-05 --margin=1 --num_negs=2 --use_case=go -ns
```
 
* FoodOn

```
python run_cat_completion.py --batch_size=8192 --emb_dim=200 --loss_type=normal --lr=0.0001 --margin=1 --num_negs=2 --use_case=foodon -ns
```

* PPI

```
python run_cat_ppi.py --batch_size=65536 --emb_dim=200 --loss_type=normal --lr=0.0001 --margin=0.1 --num_negs=2 -ns
```

## Generating your own lattice

If you have an OWL file, you can generate the lattice running the following commands:
```
cd src/
python cat_projector.py your_ontology_file.owl
```

Additionally, we have added the projection algorithm to [mOWL](https://github.com/bio-ontology-research-group/mowl). Please read the following [documentation](https://mowl.readthedocs.io/en/latest/graphs/projection.html)
   
## Citation
```
@InProceedings{10.1007/978-3-031-71167-1_19,
author="Zhapa-Camacho, Fernando
and Hoehndorf, Robert",
editor="Besold, Tarek R.
and d'Avila Garcez, Artur
and Jimenez-Ruiz, Ernesto
and Confalonieri, Roberto
and Madhyastha, Pranava
and Wagner, Benedikt",
title="Lattice-Preserving {\$}{\$}{\backslash}mathcal {\{}ALC{\}}{\$}{\$}Ontology Embeddings",
booktitle="Neural-Symbolic Learning and Reasoning",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="355--369",
isbn="978-3-031-71167-1"
}
```
