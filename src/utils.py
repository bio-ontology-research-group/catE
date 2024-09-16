import random
import numpy as np
import torch as th
import os



from org.semanticweb.owlapi.model import AxiomType as ax


subsumption_rel_name = {
    "taxonomy": "http://subclassof",
    "dl2vec": "http://subclassof",
    "onto2graph": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "owl2vec": "http://subclassof",
    "rdf": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "cat": "http://arrow",
    "cat1": "http://arrow",
    "cat2": "http://arrow",
    }

bot_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Nothing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Nothing",
    "rdf": "http://www.w3.org/2002/07/owl#Nothing",
    "cat": "owl:Nothing",
    "cat1": "owl:Nothing",
    "cat2": "owl:Nothing", 
}

top_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Thing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Thing",
    "rdf": "http://www.w3.org/2002/07/owl#Thing",
    "cat": "owl:Thing",
    "cat1": "owl:Thing",
    "cat2": "owl:Thing",
    }



prefix = {
    "pizza": "pizza",
    "dideo": "dideo",
    "fobi": "fobi",
    "nro": "nro",
    "kisao": "kisao",
    "go": "go.train",
    "foodon_comp": "foodon-merged.train",
    "go_ded": "go"
}

graph_type = {
    "taxonomy": "taxonomy",
    "onto2graph": "onto2graph",
    "owl2vec": "owl2vec",
    "rdf": "rdf",
    "cat": "cat",
    "cat1": "cat.s1",
    "cat2": "cat.s2",
    
}

suffix = {
    "taxonomy": "_no_leakage.edgelist",
    "onto2graph": "_no_leakage.edgelist",
    "owl2vec": "_no_leakage.edgelist",
    "rdf": "_no_leakage.edgelist",
    "cat": "_no_leakage.edgelist",
    "cat1": "_no_leakage.edgelist",
    "cat2": "_no_leakage.edgelist",
    
}

suffix_unsat = {
    "cat": "_initial_terminal_no_leakage.edgelist.bk",
    "cat1": "_initial_terminal_no_leakage.edgelist",
    "cat2": "_initial_terminal_no_leakage.edgelist",
    
}


suffix_completion = {
    "taxonomy": "_no_leakage.edgelist",
    "onto2graph": "_no_leakage.edgelist",
    "owl2vec": "_no_leakage.edgelist",
    "rdf": "_no_leakage_no_trivial.edgelist",
    "cat": "_no_leakage_no_trivial.edgelist",
    "cat1": "_no_leakage_no_trivial.edgelist",
    "cat2": "_no_leakage_no_trivial.edgelist",
    
}


suffix_ppi = {
    "cat": "_no_trivial.edgelist",
    "cat1": "_no_trivial.edgelist",
    "cat2": ".edgelist",
    
}



    
def pairs(iterable):
    num_items = len(iterable)
    power_set = list(powerset(iterable))
    product_set = list(product(power_set, power_set))

    curated_set = []
    for i1, i2 in product_set:
        if i1 == i2:
            continue
        if len(i1) + len(i2) != num_items:
            continue
        if len(i1) == 0 or len(i1) == num_items:
            continue
        if len(i2) == 0 or len(i2) == num_items:
            continue
        curated_set.append((i1, i2))

    return curated_set


IGNORED_AXIOM_TYPES = [ax.ANNOTATION_ASSERTION,
                       ax.ASYMMETRIC_OBJECT_PROPERTY,
                       ax.DECLARATION,
                       ax.EQUIVALENT_OBJECT_PROPERTIES,
                       ax.FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_OBJECT_PROPERTIES,
                       ax.IRREFLEXIVE_OBJECT_PROPERTY,
                       ax.OBJECT_PROPERTY_DOMAIN,
                       ax.OBJECT_PROPERTY_RANGE,
                       ax.REFLEXIVE_OBJECT_PROPERTY,
                       ax.SUB_PROPERTY_CHAIN_OF,
                       ax.SUB_ANNOTATION_PROPERTY_OF,
                       ax.SUB_OBJECT_PROPERTY,
                       ax.SWRL_RULE,
                       ax.SYMMETRIC_OBJECT_PROPERTY,
                       ax.TRANSITIVE_OBJECT_PROPERTY
                       ]
