import sys
import os
import pandas as pd
import mowl
mowl.init_jvm("10g")

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import PathDataset
from java.util import HashSet
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat


org_to_id = {'yeast': '4932', 'human': '9606'}

organism = sys.argv[1]
org_id = org_to_id[organism]

root = "../../use_cases/ppi/data/"
assert os.path.exists(root)

raw_data_path = os.path.join(root, "raw_data")
assert os.path.exists(raw_data_path)

adapter = OWLAPIAdapter()


go_annotations = pd.read_csv(os.path.join(raw_data_path, f"train/{org_id}.go.annotation.txt"), sep='\t', header=None)
go_annotations.columns = ['prot_id', 'go_id']
go_annotations['prot_id'] = go_annotations["prot_id"].apply(lambda x: adapter.create_class("http://" + x))
go_annotations['go_id'] = go_annotations['go_id'].apply(lambda x: adapter.create_class("http://purl.obolibrary.org/obo/" + x.replace(":", "_")))

ppis = pd.read_csv(os.path.join(raw_data_path, f"train/{org_id}.protein.links.v11.0.txt"), sep='\t', header=None)
ppis.columns = ['prot_id1', 'prot_id2']
ppis['prot_id1'] = ppis["prot_id1"].apply(lambda x: adapter.create_class("http://" + x))
ppis['prot_id2'] = ppis["prot_id2"].apply(lambda x: adapter.create_class("http://" + x))


go_annots_axioms = []
has_function_rel = adapter.create_object_property("http://has_function")
for i, row in go_annotations.iterrows():
    prot_id = row['prot_id']
    go_id = row['go_id']
    
    some_values_from = adapter.create_object_some_values_from(has_function_rel, go_id)
    axiom = adapter.create_subclass_of(prot_id, some_values_from)
    go_annots_axioms.append(axiom)

ppi_axioms = []
interacts_with_rel = adapter.create_object_property("http://interacts")
for i, row in ppis.iterrows():
    prot_id1 = row['prot_id1']
    prot_id2 = row['prot_id2']
    
    some_values_from = adapter.create_object_some_values_from(interacts_with_rel, prot_id2)
    axiom1 = adapter.create_subclass_of(prot_id1, some_values_from)

    some_values_from = adapter.create_object_some_values_from(interacts_with_rel, prot_id1)
    axiom2 = adapter.create_subclass_of(prot_id2, some_values_from)
    
    ppi_axioms.append(axiom1)
    ppi_axioms.append(axiom2)

dataset = PathDataset(f"{raw_data_path}/go.owl")
ontology = dataset.ontology

manager = adapter.owl_manager
axioms_set = HashSet()
axioms_set.addAll(go_annots_axioms)
axioms_set.addAll(ppi_axioms)

manager.addAxioms(ontology, axioms_set)

output_file = f"{root}/yeast.owl"
manager.saveOntology(ontology, OWLXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(output_file)))
