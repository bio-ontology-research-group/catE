import mowl
mowl.init_jvm("10g")

import subprocess
import os
import click as ck
import time
import tqdm
import rdflib
import os


def owl2rdf(owlfile):
    start_time = time.time()
    
    g = rdflib.Graph()
    g.parse (owlfile, format='application/rdf+xml')

    with open(owlfile.replace('.rdfxml', '.edgelist'), 'w') as f:
        for s, p, o in tqdm.tqdm(g, total=len(g)):
            if isinstance(s, rdflib.term.Literal):
                continue
            if isinstance(o, rdflib.term.Literal):
                continue

            f.write(str(s) + '\t' + str(p) + '\t' + str(o) + '\n')

    print("--- %s seconds ---" % (time.time() - start_time))


def onto2graph_projector(input_ontology, jar_dir):
    owlfile = os.path.abspath(input_ontology)
    
    if not owlfile.endswith('.owl'):
        raise Exception('File must be an OWL file')

    rdfxmlfile = input_ontology.replace('.owl', '.onto2graph')

    jarfile = os.path.abspath(jar_dir + 'Onto2Graph/target/Onto2Graph-1.0.jar')
    command = ['java', '-jar', jarfile, '-ont', owlfile, '-out', rdfxmlfile, '-eq', "true", "-op", "[*]", '-r', 'ELK', '-f', 'RDFXML', '-nt', '8']
    
    rdfxmlfile = rdfxmlfile + '.rdfxml'

    print("Running Onto2Graph")
    result = subprocess.run(command, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    print("Onto2Graph finished")
    print("Converting to edgelist")
    owl2rdf(rdfxmlfile)
    

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
@ck.option("--jar_dir", "-j", type=ck.Path(exists=True), required=True)
def main(input_ontology, jar_dir):
    
    onto2graph_projector(input_ontology,jar_dir)
    print("Done")

if __name__ == '__main__':
    
    
    main()
