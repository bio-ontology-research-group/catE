"""Gets all the synonym classes from an ontology"""

import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
import sys
import os
from java.util import HashSet
from org.semanticweb.owlapi.model.parameters import Imports
ontology_file = sys.argv[1]
root = os.path.dirname(ontology_file)
output_file = os.path.join(root, "synonym_classes.txt")

adapter = OWLAPIAdapter()
manager = adapter.owl_manager
factory = adapter.data_factory
ds = PathDataset(ontology_file)
ont = ds.ontology

hasExactSynonym = factory.getOWLObjectProperty(IRI.create("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"));
hasRelatedSynonym = factory.getOWLObjectProperty(IRI.create("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"));

synonyms = set()
  
for cls in ont.getClassesInSignature():
    synonyms = HashSet()
    axioms = ont.getAxioms(cls, Imports.fromBoolean(True))
    annotations1 = EntitySearcher.getAnnotations(cls, ont, hasExactSynonym)
    annotations2 = EntitySearcher.getAnnotations(cls, ont, hasRelatedSynonym)

    if len(axioms) == 0 and (len(annotations1) > 0 or len(annotations2) >0):
        synonyms.add(cls.toStringID())

with open(output_file, "w") as f:
    for synonym in synonyms:
        f.write(f"{synonym}\n")

print(f"Synonym classes written to {output_file}")
