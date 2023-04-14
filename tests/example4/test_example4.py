from unittest import TestCase
from src.cat import CategoricalProjector
import pandas as pd
import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
import os
print(os.getcwd())

example4_file = "tests/example4/fixtures/example4.owl"
example4_ground_truth = "tests/example4/fixtures/ground_truth.csv"
example4_sat_clases = "tests/example4/fixtures/satisfiable.csv"
log_file = "tests/example4/fixtures/log.csv"
c0 = "http://mowl/A or http://mowl/B and http://mowl/C or http://mowl/D and http://mowl/E or http://mowl/F"

class TestExample4(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = PathDataset(example4_file)
        self.ground_truth = pd.read_csv(example4_ground_truth, header=None)
        self.ground_truth.columns = ["source", "target"]
                
    def test_example4(self):
        """This test that the first 9 ground truth edges are in the projection"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        for head, tail in self.ground_truth.values[:4]:
            self.assertIn((head, tail), edges)

    def test_example4_sat_1(self):
        """This test that all the ground truth edges are in the projection after 1 step of saturation"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        for i in range(1):
            graph.saturate()
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        nodes = set([x[0] for x in graph_str]) | set([x[2] for x in graph_str])
        self.assertIn(c0, nodes)

        for head, tail in self.ground_truth.values[:17]:
            self.assertIn((head, tail), edges)

        
