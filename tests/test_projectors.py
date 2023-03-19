from unittest import TestCase
import wget
from src.projectors.taxonomy_projector import taxonomy_projector
from src.projectors.dl2vec_projector import dl2vec_projector
from src.projectors.owl2vecstar_projector import owl2vecstar_projector
from src.projectors.onto2graph_projector import onto2graph_projector
from src.projectors.rdf_projector import rdf_projector


import os
from time import sleep

class TestProjectors(TestCase):

    @classmethod
    def setUpClass(self):
        if not os.path.exists("pizza.owl"):
            wget.download('https://protege.stanford.edu/ontologies/pizza/pizza.owl')
        
    def test_taxonomy_projector(self):
        """This tests the taxonomy projector"""

        taxonomy_projector('pizza.owl')
        self.assertTrue(os.path.exists('pizza.taxonomy.edgelist'))


    def test_dl2vec_projector(self):
        """This tests the dl2vec projector"""

        dl2vec_projector('pizza.owl')
        self.assertTrue(os.path.exists('pizza.dl2vec.edgelist'))

    def test_owl2vecstar_projector(self):
        """This tests the owl2vecstar projector"""

        owl2vecstar_projector('pizza.owl')
        self.assertTrue(os.path.exists('pizza.owl2vec.edgelist'))

    def test_onto2graph_projector(self):
        """This tests the onto2graph projector"""

        onto2graph_projector('pizza.owl', "src/projectors/")
        self.assertTrue(os.path.exists('pizza.onto2graph.edgelist'))

    def test_rdf2vec_projector(self):
        """This tests the rdf projector"""

        rdf_projector('pizza.owl')
        self.assertTrue(os.path.exists('pizza.rdf.edgelist'))
        
