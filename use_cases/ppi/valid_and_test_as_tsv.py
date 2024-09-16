import mowl
mowl.init_jvm("10g")
from mowl.projection import TaxonomyWithRelationsProjector
from mowl.datasets import PathDataset

train_dataset = PathDataset("data/ontology.owl")
valid_dataset = PathDataset("data/valid.owl")
test_dataset = PathDataset("data/test.owl")

projector = TaxonomyWithRelationsProjector(taxonomy=False, relations=["http://interacts_with"])

train_edges = projector.project(train_dataset.ontology)
valid_edges = projector.project(valid_dataset.ontology)
test_edges = projector.project(test_dataset.ontology)

with open("data/train.tsv", "w") as f:
    for edge in train_edges:
        src, dst = edge.src, edge.dst
        src = src.split("/")[-1]
        dst = dst.split("/")[-1]
        f.write(f"{src}\t{dst}\n")

with open("data/valid.tsv", "w") as f:
    for edge in valid_edges:
        src, dst = edge.src, edge.dst
        src = src.split("/")[-1]
        dst = dst.split("/")[-1]
        f.write(f"{src}\t{dst}\n")

with open("data/test.tsv", "w") as f:
    for edge in test_edges:
        src, dst = edge.src, edge.dst
        src = src.split("/")[-1]
        dst = dst.split("/")[-1]
        f.write(f"{src}\t{dst}\n")

print("Done!")
