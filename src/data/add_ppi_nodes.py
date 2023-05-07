import mowl
mowl.init_jvm("10g")
import sys
import pandas as pd

from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP
import os

from java.util import HashSet
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat


import tqdm

owl_file = sys.argv[1]
organism = sys.argv[2]

if organism == "yeast":
    org_id  = "4932."
elif organism == "human":
    org_id = "9606."

if not owl_file.endswith(".owl"):
    raise ValueError("OWL file must end with .owl")

#get parent directory
base = os.path.dirname(owl_file)

#train_interactions = os.path.join(base, "train.tsv")
#valid_interactions = os.path.join(base, "valid.tsv")
#test_interactions = os.path.join(base, "test.tsv")


adapter = OWLAPIAdapter()
owl_manager = adapter.owl_manager
data_factory = adapter.data_factory


# Load the ontology
dataset = PathDataset(owl_file)
ontology = dataset.ontology


# Load the interactions
#train_interactions = pd.read_csv(train_interactions, sep="\t")
#train_interactions.columns = ["head", "tail"]
#valid_interactions = pd.read_csv(valid_interactions, sep="\t")
#valid_interactions.columns = ["head", "tail"]
#test_interactions = pd.read_csv(test_interactions, sep="\t")
#test_interactions.columns = ["head", "tail"]#

#all_interactions = train_interactions
#all_interactions = pd.concat([train_interactions, valid_interactions, test_interactions])

classes = ontology.getClassesInSignature()
classes = [str(c.toStringID()) for c in classes]
proteins = [c for c in classes if org_id in c]
nodes = proteins
print("Found {} proteins".format(len(proteins)))


class_nodes = [adapter.create_class(node) for node in nodes]
relation = adapter.create_object_property("http://interacts")
existential_nodes = [data_factory.getOWLObjectSomeValuesFrom(relation, class_node) for class_node in class_nodes]

bot_class = adapter.create_class(BOT)
top_class = adapter.create_class(TOP)


bot_class_axioms = [data_factory.getOWLSubClassOfAxiom(bot_class, class_node) for class_node in class_nodes]
bot_ex_axioms = [data_factory.getOWLSubClassOfAxiom(bot_class, existential_node) for existential_node in existential_nodes]

top_class_axioms = [data_factory.getOWLSubClassOfAxiom(class_node, top_class) for class_node in class_nodes]
top_ex_axioms = [data_factory.getOWLSubClassOfAxiom(existential_node, top_class) for existential_node in existential_nodes]

axioms = bot_class_axioms + bot_ex_axioms + top_class_axioms + top_ex_axioms

java_set = HashSet()
java_set.addAll(axioms)

owl_manager.addAxioms(ontology, java_set)

output_file = owl_file.replace(".owl", "_extended.owl")
owl_manager.saveOntology(ontology, OWLXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(output_file)))
