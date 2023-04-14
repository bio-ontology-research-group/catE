import mowl
mowl.init_jvm("10g")

from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.owlapi.model import IRI
from mowl.owlapi.defaults import BOT, TOP

import sys
import os

adapter = OWLAPIAdapter()
manager = adapter.owl_manager

class_a = adapter.create_class("http://mowl/A")
class_b = adapter.create_class("http://mowl/B")
class_c = adapter.create_class("http://mowl/C")
class_d = adapter.create_class("http://mowl/D")
class_e = adapter.create_class("http://mowl/E")
class_f = adapter.create_class("http://mowl/F")
bot = adapter.create_class(BOT)
top = adapter.create_class(TOP)

def example3():
    """Example 3"""
        
    ontology = manager.createOntology()
    
    not_c = adapter.create_complement_of(class_c)
    relation = adapter.create_object_property("http://mowl/R")

    exists_r_c = adapter.create_object_some_values_from(relation, class_c)
    forall_r_d = adapter.create_object_all_values_from(relation, class_d)

    axiom1 = adapter.create_subclass_of(class_a, exists_r_c)
    axiom2 = adapter.create_subclass_of(class_a, forall_r_d)
    axiom3 = adapter.create_subclass_of(class_d, not_c)

    manager.addAxiom(ontology, axiom1)
    manager.addAxiom(ontology, axiom2)
    manager.addAxiom(ontology, axiom3)

    manager.saveOntology(ontology, IRI.create("file://" + os.path.abspath("example3.owl")))

def example4():
    """Example 4"""
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    ontology = manager.createOntology()

    a_and_c = adapter.create_object_intersection_of(class_a, class_c)
    a_and_d = adapter.create_object_intersection_of(class_a, class_d)
    b_and_c = adapter.create_object_intersection_of(class_b, class_c)
    b_and_d = adapter.create_object_intersection_of(class_b, class_d)
    
    axiom1 = adapter.create_subclass_of(a_and_c, bot)
    axiom2 = adapter.create_subclass_of(a_and_d, bot)
    axiom3 = adapter.create_subclass_of(b_and_c, bot)
    axiom4 = adapter.create_subclass_of(b_and_d, bot)

    manager.addAxiom(ontology, axiom1)
    manager.addAxiom(ontology, axiom2)
    manager.addAxiom(ontology, axiom3)
    manager.addAxiom(ontology, axiom4)

    
    # Query axiom

    a_or_b = adapter.create_object_union_of(class_a, class_b)
    c_or_d = adapter.create_object_union_of(class_c, class_d)
    e_or_f = adapter.create_object_union_of(class_e, class_f)

    c0 = adapter.create_object_intersection_of(a_or_b, c_or_d, e_or_f)

    query_axiom = adapter.create_subclass_of(c0, top)
    manager.addAxiom(ontology, query_axiom)

    manager.saveOntology(ontology, IRI.create("file://" + os.path.abspath("example4.owl")))

if __name__ == "__main__":
    num = sys.argv[1]

    if num == "3":
        example3()
    elif num == "4":
        example4()
