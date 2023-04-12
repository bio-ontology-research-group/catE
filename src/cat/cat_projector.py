import mowl
mowl.init_jvm("10g")

import jpype
@jpype.JImplementationFor('java.lang.Comparable')
class _JComparableHash(object):
    def __hash__(self):
        return self.hashCode()

import click as ck
import sys
sys.path.append('../..')

from edge import Edge, Node
from utils import IGNORED_AXIOM_TYPES, pairs

from mowl.owlapi import OWLAPIAdapter, OWLClass
from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi.constants import MOWL_IRI as mowl_iri
from mowl.datasets import PathDataset

from org.semanticweb.owlapi.model import OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLObjectProperty, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectUnionOf,\
    OWLObjectExactCardinality, OWLDataHasValue, OWLDataSomeValuesFrom, OWLObjectMinCardinality, \
    OWLObjectHasSelf, AxiomType, OWLEntity, OWLObjectHasValue, OWLObjectOneOf, OWLClassExpression, \
    OWLDataExactCardinality, OWLDataMinCardinality, OWLDataMaxCardinality, OWLDataOneOf, EntityType, \
    OWLObjectInverseOf, OWLObjectMaxCardinality, OWLDataAllValuesFrom

from org.semanticweb.owlapi.model import ClassExpressionType as CT

from org.semanticweb.owlapi.model.parameters import Imports

import java
from java.util import HashSet

from tqdm import tqdm
import logging
import networkx as nx



from tqdm import tqdm
import random

adapter = OWLAPIAdapter()

top_node = Node(owl_class = adapter.create_class(TOP))
bot_node = Node(owl_class = adapter.create_class(BOT))

class Graph():
    def __init__(self):
        self._node_to_id = {}
        self._id_to_node = {}
        self._out_edges = dict()
        self._in_edges = dict()

    
    @property
    def node_to_id(self):
        return self._node_to_id

    @property
    def id_to_node(self):
        return self._id_to_node
    
    @property
    def nodes(self):
        return set(self.node_to_id.keys())

    @property
    def out_edges(self):
        return self._out_edges

    @property
    def in_edges(self):
        return self._in_edges

    def add_node(self, node):

        if node in self.node_to_id:
            return
        
        if not isinstance(node, Node):
            raise TypeError(f"Node must be of type Node. Got {type(node)}")
        if bot_node not in self.node_to_id:
            self.node_to_id[bot_node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[bot_node]] = bot_node
            self.out_edges[bot_node] = set()
            self.in_edges[bot_node] = set()

        if top_node not in self.node_to_id:
            self.node_to_id[top_node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[top_node]] = top_node
            self.out_edges[top_node] = set()
            self.in_edges[top_node] = set()

            
        if not node in self.node_to_id:
            self.node_to_id[node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[node]] = node
            self.out_edges[node] = set()
            self.in_edges[node] = set()

        self.in_edges[node].add(node)
        self.out_edges[node].add(node)

        self.in_edges[node].add(bot_node)
        self.out_edges[bot_node].add(node)

        self.in_edges[top_node].add(node)
        self.out_edges[node].add(top_node)


                    
        if node.in_object_category() and not node.domain and not node.codomain and node.owl_class.getClassExpressionType() == CT.OWL_CLASS:
            negated = node.negate()
            and_node = adapter.create_object_intersection_of(node.owl_class, negated.owl_class)
            and_node = Node(owl_class = and_node)
            self.add_edge(Edge(and_node, "http://arrow", bot_node))
            self.add_edge(Edge(and_node, "http://arrow", node))
            self.add_edge(Edge(and_node, "http://arrow", negated))

            or_node = adapter.create_object_union_of(node.owl_class, negated.owl_class)
            or_node = Node(owl_class = or_node)
            self.add_edge(Edge(top_node, "http://arrow", or_node))
            self.add_edge(Edge(node, "http://arrow", or_node))
            self.add_edge(Edge(negated, "http://arrow", or_node))
            
        self.add_node(node.nnf())
                                        
            
    def add_edge(self, edge):
        src = edge.src
        dst = edge.dst
        self.add_node(src)
        self.add_node(dst)
                                                    
        self.out_edges[src].add(dst)
        self.in_edges[dst].add(src)

    def add_all_edges(self, *edges):
        for edge in edges:
            self.add_edge(edge)

    def as_edgelist(self):
        edges = []
        for source, targets in self.out_edges.items():
            for target in targets:
                edges.append((source, target))
        return edges

    def as_str_edgelist(self):
        edges = []
        for source, targets in self.out_edges.items():
            source = str(source)
            for target in targets:
                target = str(target)
                edges.append((source, "http://arrow", target))
        return edges

    def as_edges(self):
        for source, targets in self.out_edges.items():
            for target in targets:
                yield Edge(source, "http://arrow", target)

    def _lemma_6(self):
        # First equation, nothing to do
        # Second equation and right side of third equation and right side of fourth equation
        negated_nodes = set()
        for node in self.nodes:
            if node.is_negated():
                #print("Negated node: {}\t{}".format(node, node.owl_class))
                negated_nodes.add(node)
            
        for neg_node in negated_nodes:
            in_edges = list(self.in_edges[neg_node])
            for in_node in in_edges:
                node = neg_node.get_operand()
                #node = Node(owl_class = node)
                neg_in_node = in_node.negate()
                #neg_in_node = adapter.create_complement_of(in_node.owl_class)
                #neg_in_node = Node(owl_class = neg_in_node)
                op_edge = Edge(node, "saturation_lemma6", neg_in_node)
                
                intersection = adapter.create_object_intersection_of(node.owl_class, in_node.owl_class)
                intersection = Node(owl_class = intersection)
                edge_int = Edge(intersection, "saturation_lemma6", bot_node)

                union = adapter.create_object_union_of(node.owl_class, in_node.owl_class)
                union = Node(owl_class = union)
                edge_un = Edge(top_node, "saturation_lemma6", union)

                self.add_all_edges(op_edge, edge_int, edge_un)

                #TODO last equation (Morgan law)

        # Left side of third equation
        intersection_nodes = set()
        
        for node in self.nodes:
            if node.is_intersection():
                intersection_nodes.add(node)

        for int_node in intersection_nodes:
            if not bot_node in self.out_edges[int_node]:
                continue

            operands = list(int_node.owl_class.getOperandsAsList())
            intersection_pairs = pairs(operands)
            
            for node1, node2 in intersection_pairs:
                if len(node1) > 1:
                    node1 = adapter.create_object_intersection_of(*node1)
                else:
                    node1 = node1[0]
                node1 = Node(owl_class = node1)
                if len(node2) > 1:
                    node2 = adapter.create_object_intersection_of(*node2)
                else:
                    node2 = node2[0]
                node2 = Node(owl_class = node2.getObjectComplementOf())
                edge = Edge(node1, "saturation_lemma6", node2)
                self.add_edge(edge)
                
        # Left side of fourth equation
        union_nodes = set()
        for node in self.nodes:
            if node.is_union():
                union_nodes.add(node)

        for un_node in union_nodes:
            if not top_node in self.in_edges[un_node]:
                continue

            operands = list(un_node.owl_class.getOperandsAsList())
            union_pairs = pairs(operands)
            for node1, node2 in union_pairs:
                if len(node1) > 1:
                    node1 = adapter.create_object_union_of(*node1)
                else:
                    node1 = node1[0]
                node1 = Node(owl_class = node1.getObjectComplementOf())

                if len(node2) > 1:
                    node2 = adapter.create_object_union_of(*node2)
                else:
                    node2 = node2[0]
                    
                node2 = Node(owl_class = node2)

                edge = Edge(node1, "saturation_lemma6", node2)
                self.add_edge(edge)
                        


                            
    def _definition_7(self):
        #Last equation, other equations are covered in lemma 8
        relations = set()
        for node in self.nodes:
            if node.is_whole_relation():
                relations.add(node)

        for rel in relations:
            #print("\tWhole relation: {}".format(rel))
            for in_rel in self.in_edges[rel]:
                if not in_rel.in_relation_category():
                    continue
                in_rel_codomain = in_rel.to_codomain()
                if not in_rel_codomain in self.nodes:
                    continue

                #print(f"\t\tIn relation: {in_rel}")
                #print(f"\t\tIn relation codomain: {in_rel_codomain}")
                for cod in self.out_edges[in_rel_codomain]:
                    if cod.domain or cod.codomain or cod.in_relation_category():
                        continue
                    #print(f"\t\t\tCodomain: {cod}")
                    in_rel_domain = in_rel.to_domain()

                    ex_rel_cod = adapter.create_object_some_values_from(rel.relation, cod.owl_class)
                    node = Node(owl_class = ex_rel_cod, relation = rel.relation, domain = True)
                    edge = Edge(in_rel_domain, "saturation_definition7", node)
                    #print(f"\t\t\t\tAdding edge: {edge.src} -> {edge.dst}")
                    self.add_edge(edge)
                    
                
                    

                
    def _lemma_8(self):
        existential_nodes = set()
        for node in self.nodes:
            if node.is_existential():
                existential_nodes.add(node)

        for node in existential_nodes:
            filler = node.owl_class.getFiller()
            property_ = node.owl_class.getProperty()
            filler_node = Node(owl_class = filler)
            domain_node = Node(owl_class = node.owl_class, relation = property_, domain = True)


            in_edges = list(self.in_edges[filler_node])
            for in_filler in in_edges:
                ex_in_class = adapter.create_object_some_values_from(node.owl_class.getProperty(), in_filler.owl_class)
                ex_in_node = Node(owl_class = ex_in_class)
                edge = Edge(ex_in_node, "saturation_lemma8", node)
                self.add_edge(edge)

                domain_in_node = Node(owl_class=ex_in_class, relation = property_, domain = True)
                edge = Edge(domain_in_node, "saturation_lemma8", domain_node)
                self.add_edge(edge)
                edge = Edge(domain_in_node, "saturation_lemma8", ex_in_node)
                self.add_edge(edge)
                edge = Edge(ex_in_node, "saturation_lemma8", domain_in_node)
                self.add_edge(edge)

            out_edges = list(self.out_edges[filler_node])
            for out_filler in out_edges:
                if out_filler.owl_class is None:
                    continue
                if out_filler.is_owl_nothing():
                    continue
                    edge = Edge(node, "saturation_lemma8", bot_node)
                    self.add_edge(edge)
                else:
                    ex_out_class = adapter.create_object_some_values_from(node.owl_class.getProperty(), out_filler.owl_class)
                    ex_out_node = Node(owl_class = ex_out_class)
                    edge = Edge(node, "saturation_lemma8", ex_out_node)
                    self.add_edge(edge)

                    domain_out_node = Node(owl_class=ex_out_class, relation = property_, domain = True)
                    edge = Edge(domain_node, "saturation_lemma8", domain_out_node)
                    self.add_edge(edge)
                    edge = Edge(domain_out_node, "saturation_lemma8", ex_out_node)
                    self.add_edge(edge)
                    edge = Edge(ex_out_node, "saturation_lemma8", domain_out_node)
                    self.add_edge(edge)

        
    def as_nx(self):
        logging.debug("Converting to networkx")
        G = nx.DiGraph()
        G.add_nodes_from(list(self.node_to_id.values()))
        for edge in self.as_edgelist():
            src, dst = edge
                            
            if src.in_object_category() and dst.in_object_category():
                if False and src == top_node or src == bot_node or dst == top_node or dst == bot_node:
                    continue
                G.add_edge(self.node_to_id[src], self.node_to_id[dst])
        logging.debug("Done converting to networkx")
        return G
                
    def transitive_closure(self):
        logging.debug("Computing transitive closure")
        G = self.as_nx()
        G = nx.transitive_closure(G)
        logging.debug("Done computing transitive closure in NetworkX")
        for src, dst in tqdm(G.edges()):
            src = self.id_to_node[src]
            dst = self.id_to_node[dst]
            self.add_edge(Edge(src, "transitive_closure", dst))
        logging.debug("Done computing transitive closure")

                
    def saturate(self):
        self._lemma_6()
        self._definition_7()
        self._lemma_8()

    def is_unsatisfiable(self, node):
        if not isinstance(node, Node):
            raise TypeError("node must be of type Node")
        if node not in self.nodes:
            raise ValueError("Node is not in graph")
        out_edges = self.out_edges[node]
        if bot_node in out_edges:
            return True
        else:
            return False

    def get_unsatisfiable_nodes(self):

        def is_trivial_unsat(node):
            trivial = False
            if node.domain or node.codomain or node.in_relation_category():
                return trivial

            owl_class = node.owl_class
            if owl_class.getClassExpressionType() == CT.OBJECT_INTERSECTION_OF:
                ops = owl_class.getOperandsAsList()
                if len(ops) == 2:
                    op1, op2 = tuple(ops)
                    if op2.getClassExpressionType() == CT.OBJECT_COMPLEMENT_OF:
                        op2 = op2.getOperand()
                        if op1.equals(op2):
                            trivial = True
                    if op1.getClassExpressionType() == CT.OBJECT_COMPLEMENT_OF:
                        op1 = op1.getOperand()
                        if op2.equals(op1):
                            trivial = True
            return trivial
        
        unsat = set()
        for node in self._in_edges[bot_node]:
            if is_trivial_unsat(node):
                continue
            unsat.add(node)
            
        return unsat


    def write_to_file(self, outfile):
        with open(outfile, "w") as f:
            edges = self.as_str_edgelist()
            for src, rel, dst in tqdm(edges):
                f.write(f"{src}\t{rel}\t{dst}\n")

class CategoricalProjector():
    """This class implements the projection of OWL axioms into a graph using categorical diagrams.
 
    """

    
    def __init__(self):

        self.adapter = OWLAPIAdapter()
        self.ont_manager = self.adapter.owl_manager
        self.data_factory = self.adapter.data_factory

        self.graph = Graph()


    def project(self, ontology, identity = False, composition = False):
        """Project an ontology into a graph using categorical diagrams."""

        all_classes = ontology.getClassesInSignature()
        for cls in tqdm(all_classes):
            self.graph.add_node(Node(owl_class = cls))
        all_axioms = ontology.getAxioms(True)
        
        for axiom in tqdm(all_axioms, total = len(all_axioms)):
            self.graph.add_all_edges(*list(self.process_axiom(axiom)))

            
        if identity:
            raise NotImplementedError("Identity projection not implemented yet.")

        if composition:
            raise NotImplementedError("Composition projection not implemented yet.")
        
        return self.graph.as_edges()
    def process_axiom(self, axiom):
        """Process an OWLClass and return a list of edges."""

        axiom_type = axiom.getAxiomType()
        if axiom_type == AxiomType.SUBCLASS_OF:
            return self.process_subclassof(axiom)
        elif axiom_type == AxiomType.EQUIVALENT_CLASSES:
            return self.process_equivalentclasses(axiom)
        elif axiom_type == AxiomType.DISJOINT_CLASSES:
            return self.process_disjointness(axiom)
        elif axiom_type == AxiomType.CLASS_ASSERTION:
            return []
            #return self.process_class_assertion(axiom)
        elif axiom_type == AxiomType.OBJECT_PROPERTY_ASSERTION:
            return []
            #return self.process_object_property_assertion(axiom)
        elif axiom_type in IGNORED_AXIOM_TYPES:
            #Ignore these types of axioms
            return []
        else:
            print(f"process_axiom: Unknown axiom type: {axiom_type}")
            return []


    def process_equivalentclasses(self, axiom):
        """Process an EquivalentClasses axiom and return a list of edges."""

        subclass_axioms = axiom.asOWLSubClassOfAxioms()
        edges = set()
        for subclass_axiom in subclass_axioms:
            edges |= self.process_subclassof(subclass_axiom)
        
        if edges == set():
            print(f"No edges found for EquivalentClasses axiom: {axiom}")
        return edges

    def process_disjointness(self, axiom):
        """Process a disjointness axiom"""
        subclass_axioms = axiom.asOWLSubClassOfAxioms()
        edges = set()
        for subclass_axiom in subclass_axioms:
            edges |= self.process_subclassof(subclass_axiom)
        
        return edges

    def process_class_assertion(self, axiom):
        """Process a class assertion axiom"""
        individual = axiom.getIndividual()
        class_expression = axiom.getClassExpression()

        class_virtual_node, edges = self.process_expression_and_get_virtual_node(class_expression)
        edges.add(Edge(str(individual.toStringID()), "http://type", class_virtual_node))
        return edges

    def process_object_property_assertion(self, axiom):
        """Process an object property assertion axiom"""
        subject = axiom.getSubject()
        object = axiom.getObject()
        property = axiom.getProperty()

        edges = set()
        edges.add(Edge(subject.toStringID(), property.toStringID(), object.toStringID()))
        return edges
        
    def process_subclassof(self, axiom):
        """Process a SubClassOf axiom and return a list of edges."""
        sub_class = axiom.getSubClass()
        super_class = axiom.getSuperClass()
        sub_edges = set()
        super_edges = set()

        sub_node, sub_edges = self.process_expression_and_get_complex_node(sub_class)
        #print(axiom)
        super_node, super_edges = self.process_expression_and_get_complex_node(super_class)
                                
        if (sub_node is None) or (super_node is None):
            return set()

        edges = set()
        edges.add(self.subsumption_arrow(sub_node, super_node))
        edges |= sub_edges
        edges |= super_edges

        not_sub_class = self.adapter.create_complement_of(sub_class)
        union = self.adapter.create_object_union_of(not_sub_class, super_class)
        union_complex_node, union_edges = self.process_expression_and_get_complex_node(union)

        
        
        edges |= union_edges
        edges.add(self.subsumption_arrow(top_node, union_complex_node))
        
        return edges


    def process_expression_and_get_complex_node(self, expression):

        expr_type = expression.getClassExpressionType()

        edges = set()
        
        if expr_type == CT.OWL_CLASS:
            return Node(owl_class=expression), edges
            
        if expr_type == CT.OBJECT_INTERSECTION_OF:
            prod_complex_node = expression
            operands = expression.getOperandsAsList()
                        
            for op in operands:
                op_complex_node, op_edges = self.process_expression_and_get_complex_node(op)
                if op_complex_node is None:
                    return None, set()

                edges |= op_edges
                edges.add(self.product_limit_arrow(prod_complex_node, op_complex_node))

            #TODO: add arrows to fulfill distributivity

            return prod_complex_node, edges
            
        elif expr_type == CT.OBJECT_UNION_OF:
            coprod_complex_node = expression
            operands = expression.getOperandsAsList()
            
            for op in operands:
                op_complex_node, op_edges = self.process_expression_and_get_complex_node(op)
                if op_edges == None:
                    return None, set()
                
                edges |= op_edges
                edges.add(self.coproduct_limit_arrow(op_complex_node,
                                                     coprod_complex_node))

                                                                            
            return coprod_complex_node, edges
                    
        elif expr_type == CT.OBJECT_SOME_VALUES_FROM:
            
            existential_complex_node = expression
                            
            property_ = expression.getProperty()

            if not isinstance(property_, OWLObjectInverseOf):
                if property_.getEntityType() != EntityType.OBJECT_PROPERTY:
                    return None, set()
            else:
                if property_.isNamed():
                    property_ = property_.asOWLObject()
                else:
                    return None, set()
                    
            filler = expression.getFiller()
            filler_info = self.process_expression_and_get_complex_node(filler)
            filler_node, filler_edges = filler_info
            if filler_node is None:
                return None, set()
            
            edges |= filler_edges


            rel_exists_r_c = Node(owl_class = expression, relation=property_)
            codomain_rel_exists_r_c = Node(owl_class=expression, relation=property_, codomain=True)
            domain_rel_exists_r_c = Node(owl_class=expression, relation=property_, domain=True)
                        
            edges.add(Edge(rel_exists_r_c, "http://general_arrow", Node(relation=property_)))
            edges.add(Edge(codomain_rel_exists_r_c, "http://general_arrow", Node(owl_class=filler)))
            
            exist_node = Node(owl_class=existential_complex_node)
            edges.add(Edge(domain_rel_exists_r_c, "http://general_arrow", exist_node))
            edges.add(Edge(exist_node, "http://general_arrow", domain_rel_exists_r_c))
                        
            return existential_complex_node, edges
            
                         
        elif expr_type == CT.OBJECT_ALL_VALUES_FROM:
            universal_complex_node = expression
                                        
            property_ = expression.getProperty()
            filler = expression.getFiller()
            not_filler = self.adapter.create_complement_of(filler)
            rel_not_filler = self.adapter.create_object_some_values_from(property_, not_filler)
            not_rel_not_filler = self.adapter.create_complement_of(rel_not_filler)

            not_rel_not_filler_info = self.process_expression_and_get_complex_node(not_rel_not_filler)
            not_rel_not_filler_node, not_rel_not_filler_edges = not_rel_not_filler_info
            if not_rel_not_filler_node is None:
                return None, set()

            filler_info = self.process_expression_and_get_complex_node(filler)
            filler_node, filler_edges = filler_info
            if filler_node is None:
                return None, set()
            
            edges |= not_rel_not_filler_edges
            edges |= filler_edges

            edges.add(self.subsumption_arrow(universal_complex_node, not_rel_not_filler_node))
            edges.add(self.subsumption_arrow(not_rel_not_filler_node, universal_complex_node))

            not_domain_rel_n_filler = Node(owl_class=rel_not_filler, relation=property_, domain=True, negated_domain=True)
            edges.add(self.subsumption_arrow(not_domain_rel_n_filler, universal_complex_node))
            edges.add(self.subsumption_arrow(universal_complex_node, not_domain_rel_n_filler))
            return universal_complex_node, edges

        elif expr_type == CT.OBJECT_COMPLEMENT_OF:
            
            negation_complex_node = expression
            operand = expression.getOperand()
            
            operand_info = self.process_expression_and_get_complex_node(operand)
            operand_node, operand_edges = operand_info

            union = self.adapter.create_object_union_of(expression, operand)
            union = Node(owl_class = union)
            intersection = self.adapter.create_object_intersection_of(expression, operand)
            intersection = Node(owl_class=intersection)
            edges |= operand_edges

                                                
            edges.add(Edge(intersection, "http://general_arrow", bot_node))
            edges.add(Edge(top_node, "http://general_arrow", union))

            return negation_complex_node, edges
                         
        elif isinstance(expression, (OWLObjectExactCardinality, OWLObjectMinCardinality, OWLObjectHasSelf, OWLObjectHasValue, OWLObjectOneOf, OWLDataExactCardinality, OWLDataMinCardinality, OWLDataHasValue, OWLDataOneOf, OWLDataSomeValuesFrom, OWLDataMaxCardinality, OWLObjectMaxCardinality, OWLDataAllValuesFrom)):
            return None, set()
        else:
            print("process expression and get complex node: Unknown super class type: {}".format(expression))
        
                            
    ####################### MORPHISMS ###########################

    # decorator that transforms params as Node
    def node_params(func):
        def wrapper(self, src, dst):
            if not isinstance(src, Node):
                src = Node(owl_class=src)
            if not isinstance(dst, Node):
                dst = Node(owl_class=dst)
            return func(self, src, dst)
        return wrapper

    @node_params
    def subsumption_arrow(self, src, dst):
        rel = "http://subsumption_arrow"
        return Edge(src, rel, dst)

    @node_params
    def coproduct_limit_arrow(self, src, dst):
        rel = "http://injects"
        return Edge(src, rel, dst)

    @node_params
    def product_limit_arrow(self, src, dst):
        rel = "http://projects"
        return Edge(src, "http://projects", dst)

    @node_params
    def coproduct_weak_arrow(self, src, dst):
        rel = "http://weak_injects"
        return Edge(src, rel, dst)

    @node_params
    def product_weak_arrow(self, src, dst):
        rel = "http://weak_projects"
        return Edge(src, rel, dst)

    @node_params
    def coproduct_factorizer_arrow(self, src, dst):
        rel = "http://coproduct_factorizer"
        return Edge(src, rel, dst)

    @node_params
    def product_factorizer_arrow(self, src, dst):
        rel = "http://product_factorizer"
        return Edge(src, rel, dst)




def add_extra_existential_axioms(ontology):
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    
    classes = ontology.getClassesInSignature()
    roles = ontology.getObjectPropertiesInSignature()
    
    bot_class = adapter.create_class(BOT)
    top_class = adapter.create_class(TOP)

    
    axioms = HashSet()
    for role in roles:
        for cls in classes:
            existential = adapter.create_object_some_values_from(role, cls)
            axiom1 = adapter.create_subclass_of(existential, top_class)
            axiom2 = adapter.create_subclass_of(bot_class, existential)
            axioms.add(axiom1)
            axioms.add(axiom2)

    print(f"Axioms before: {len(ontology.getAxioms())}")
    manager.addAxioms(ontology, axioms)
    print(f"Axioms after: {len(ontology.getAxioms())}")
            

@ck.command()
@ck.option('--input-ontology', '-i', required=True, type = ck.Path(exists=True))
@ck.option('--saturation-steps', '-s', default=0, type=int)
@ck.option('--with-transitive-reduction', '-t', is_flag=True)
@ck.option('--add-existential-axioms', '-e', is_flag=True)
def main(input_ontology, saturation_steps, with_transitive_reduction, add_existential_axioms):
        
    if not input_ontology.endswith(".owl"):
        raise Exception("The ontology file must be an OWL file")
    
    ds = PathDataset(input_ontology)

    if add_existential_axioms:
        add_extra_existential_axioms(ds.ontology)
    
    projector = CategoricalProjector()
    projector.project(ds.ontology)
    graph = projector.graph

    print("--------------")
    outfile = input_ontology.replace(".owl", ".cat.edgelist")
    
    if saturation_steps > 0:
        for s in range(saturation_steps):
            print(f"Saturation step {s+1}")
            graph.saturate()
            if with_transitive_reduction:
                graph.transitive_closure()

        outfile = input_ontology.replace(".owl", f".cat.s{s+1}.edgelist")
        if with_transitive_reduction:
            outfile = outfile.replace("edgelist", "transitive.edgelist")

    print(f"Graph computed. Writing into file: {outfile}")
    with open(outfile, "w") as f:
        edges = list(set(graph.as_str_edgelist()))
        for src, rel, dst in edges:
            f.write(f"{src}\t{rel}\t{dst}\n")

    print("Done.")
        

if __name__ == "__main__":
    main()


