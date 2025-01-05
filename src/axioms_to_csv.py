import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
import os
from tqdm import tqdm

from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import AxiomType
from org.semanticweb.owlapi.model.parameters import Imports

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    """Gets the subsumtpion axioms from an ontology of the form C subclassof D where C and D are concept names"""

    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    outfile_name = os.path.splitext(input_ontology)[0] + "_subsumption_asserted.csv"

    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    with open(outfile_name, "w") as outfile:
        for axiom in tqdm(tbox_axioms):
            if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
                continue
            
            sub = axiom.getSubClass()
            super_ = axiom.getSuperClass()
            if sub.getClassExpressionType() != CT.OWL_CLASS:
                continue
            if super_.getClassExpressionType() != CT.OWL_CLASS:
                continue
            sub_name = str(sub.toStringID())
            super_name = str(super_.toStringID())
            outfile.write(f"{sub_name},{super_name}\n")
                                                
    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
