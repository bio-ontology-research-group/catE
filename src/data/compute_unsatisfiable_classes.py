import mowl

mowl.init_jvm("2000g")

from mowl.datasets import PathDataset
import click as ck
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.HermiT import Reasoner
from tqdm import tqdm

@ck.command()
@ck.option("--input-ontology", "-i", type=ck.Path(exists=True), required=True)
@ck.option("--reasoner", "-r", type=ck.Choice(["elk", "hermit"]), required=True)
@ck.option("--output-file", "-o", type=ck.Path(exists=False), required=True)
def main(input_ontology, reasoner, output_file):
    ds = PathDataset(input_ontology)
    bot="owl:Nothing"
    if reasoner == "elk":
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(ds.ontology)
    elif reasoner == "hermit":
        reasoner = Reasoner.ReasonerFactory().createReasoner(ds.ontology)
        reasoner.precomputeInferences()

    
    for cls in tqdm(ds.ontology.getClassesInSignature()):
        if not cls.isBottomEntity() and not reasoner.isSatisfiable(cls):
            cls_str = str(cls.toStringID())
            with open(output_file, "a") as f:
                f.write(f"{cls_str},{bot}\n")
        
    #unsatisfiable_classes = reasoner.getUnsatisfiableClasses()

    #with open("unsatisfiable_classes.txt", "w") as f:
    #    for cls in unsatisfiable_classes.getEntitiesMinusBottom():
    #        f.write(str(cls.getIRI().toString()) + "\n")


if __name__ == "__main__":
    main()
