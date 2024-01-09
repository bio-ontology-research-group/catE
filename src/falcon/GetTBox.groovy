@Grab(group='org.slf4j', module='slf4j-api', version='1.7.36')
@Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3')
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.5.20')

import java.util.logging.Logger
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary
import org.semanticweb.owlapi.model.*
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.profiles.*
import org.semanticweb.owlapi.util.*
import org.semanticweb.owlapi.io.*
import org.semanticweb.elk.owlapi.*
import org.semanticweb.owlapi.model.parameters.Imports
import org.semanticweb.owlapi.search.*
import org.semanticweb.owlapi.normalform.*


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLDataFactory fac = manager.getOWLDataFactory()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(new File(args[0]))
NegationalNormalFormConverter converter = new NegationalNormalFormConverter(fac)

ont.getTBoxAxioms().each { axiom ->
    if (axiom.isOfType(AxiomType.SUBCLASS_OF)) {
		axiom = (OWLSubClassOfAxiom)axiom
		def sub = axiom.getSubClass()
		def sup = axiom.getSuperClass()
		def c = fac.getOWLObjectIntersectionOf(sub, fac.getOWLObjectComplementOf(sup))	
		if (c.toString().contains('Nothing') == false) {
			println c
		}
    } 
	else if (axiom.isOfType(AxiomType.EQUIVALENT_CLASSES)) {
		axiom = (OWLEquivalentClassesAxiom)axiom
		def s = axiom.getClassExpressions()
		def left = s[0]
		def right = s[1]
		def ret_left = fac.getOWLObjectIntersectionOf(left, fac.getOWLObjectComplementOf(right))	
		def ret_right = fac.getOWLObjectIntersectionOf(right, fac.getOWLObjectComplementOf(left))	
		println ret_left
		println ret_right
	} 
	else if (axiom.isOfType(AxiomType.DISJOINT_CLASSES)) {
		axiom = (OWLDisjointClassesAxiom)axiom
		def s = axiom.getClassExpressions()
		for (int i = 0; i < s.size(); i++) {
			for (int j = 0; j < i; j++){
				if (i != j) {
					def left = s[i]
					def right = s[j]
					def c = fac.getOWLObjectIntersectionOf(fac.getOWLObjectIntersectionOf(left, right), fac.getOWLObjectComplementOf(fac.getOWLNothing()))
					println c
				}
			}
		}
	}
}
