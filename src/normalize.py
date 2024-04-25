import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.ontology.normalize import ELNormalizer
from mowl.owlapi.adapter import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from java.util import HashSet
import sys
import os

input_file = sys.argv[1]

assert input_file.endswith(".owl"), f"Input ontology must be an OWL file, but got {input_file}"

output_file = input_file.replace(".owl", "_normalized.owl")

ds = PathDataset(input_file)
normalizer = ELNormalizer()
gcis = normalizer.normalize(ds.ontology)

adapter = OWLAPIAdapter()
manager = adapter.owl_manager

normalized_ontology = manager.createOntology()

for gci, axioms in gcis.items():
    set_axioms = HashSet()
    for axiom in axioms:
        set_axioms.add(axiom.owl_axiom)
    manager.addAxioms(normalized_ontology, set_axioms)
manager.saveOntology(normalized_ontology, OWLXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(output_file)))

