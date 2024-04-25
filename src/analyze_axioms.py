import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
import sys

from org.semanticweb.owlapi.model.parameters import Imports

input_file = sys.argv[1]
assert input_file.endswith(".owl"), f"Input file must be an OWL file, but got {input_file}"

ds = PathDataset(input_file)
ontology = ds.ontology

axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))

print(f"Number of TBox axioms: {len(axioms)}")

union_of = 0
forall = 0
complement_of = 0

for axiom in axioms:

    axiom_str = str(axiom.toString())
    
    if "ObjectUnionOf" in axiom_str:
        union_of += 1
    elif "ObjectAllValuesFrom" in axiom_str:
        forall += 1
    elif "ObjectComplementOf" in axiom_str:
        complement_of += 1
        

print(f"Number of ObjectUnionOf axioms: {union_of}")
print(f"Number of ObjectAllValuesFrom axioms: {forall}")
print(f"Number of ObjectComplementOf axioms: {complement_of}")
    

