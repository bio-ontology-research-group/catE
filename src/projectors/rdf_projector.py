import mowl
mowl.init_jvm("10g")
import click as ck
import time
import rdflib
import tqdm
import click as ck

def owl2rdf(owlfile):
    start_time = time.time()
    
    g = rdflib.Graph()
    g.parse (owlfile, format='application/rdf+xml')

    with open(owlfile.replace('.owl', '.rdf.edgelist'), 'w') as f:
        for s, p, o in tqdm.tqdm(g, total=len(g)):
            #do not consider annotations
            if p == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#annotatedSource'):
                continue
            if p == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#annotatedProperty'):
                continue
            if p == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#annotatedTarget'):
                continue
            if p == rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#comment'):
                continue
            if p == rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#label'):
                continue
            
            if isinstance(s, rdflib.term.Literal):
                continue
            
            if isinstance(o, rdflib.term.Literal):
                continue
                                                 
            f.write(str(s) + '\t' + str(p) + '\t' + str(o) + '\n')

    print("--- %s seconds ---" % (time.time() - start_time))



def rdf_projector(input_ontology):
    owlfile = input_ontology
    if not owlfile.endswith(".owl"):
        raise Exception("Input file must be an OWL file")

    owl2rdf(owlfile)


@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    rdf_projector(input_ontology)
    print("Done")

if __name__ == "__main__":
    
    main()
