from unittest import TestCase
from src.cat import CategoricalProjector
import pandas as pd
import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
import os
print(os.getcwd())

example3_file = "tests/example3/fixtures/example3.owl"
example3_ground_truth = "tests/example3/fixtures/ground_truth.csv"
example3_sat_clases = "tests/example3/fixtures/satisfiable.csv"

class TestExample3(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = PathDataset(example3_file)
        self.ground_truth = pd.read_csv(example3_ground_truth, header=None)
        self.ground_truth.columns = ["source", "target"]
        self.sat_classes = pd.read_csv(example3_sat_clases, header=None)
        self.sat_classes.columns = ["source", "target"]

        
    def test_example3(self):
        """This test that the first 9 ground truth edges are in the projection"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        for head, tail in self.ground_truth.values[:12]:
            self.assertIn((head, tail), edges)

        for head, tail in self.sat_classes.values:
            self.assertNotIn((head, tail), edges)
            
    def test_example3_sat_1(self):
        """This test that all the ground truth edges are in the projection after 1 step of saturation"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        for i in range(1):
            graph.saturate()
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        for head, tail in self.ground_truth.values[:13]:
            self.assertIn((head, tail), edges)

        for head, tail in self.sat_classes.values:
            self.assertNotIn((head, tail), edges)

    def test_example3_sat_2(self):
        """This test that all the ground truth edges are in the projection after 1 step of saturation"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        for i in range(2):
            graph.saturate()
            graph.transitive_closure()
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        for head, tail in self.ground_truth.values.tolist()[:16]:
            self.assertIn((head, tail), edges)

        for head, tail in self.sat_classes.values:
            self.assertNotIn((head, tail), edges)

    def test_example3_sat_3(self):
        """This test that all the ground truth edges are in the projection after 1 step of saturation"""
        cat = CategoricalProjector()
        cat.project(self.dataset.ontology)
        graph = cat.graph
        for i in range(3):
            graph.saturate()
            graph.transitive_closure()
        graph_str = graph.as_str_edgelist()
        edges = set([(x[0], x[2]) for x in graph_str])

        for head, tail in self.ground_truth.values:
            self.assertIn((head, tail), edges)

        for head, tail in self.sat_classes.values:
            self.assertNotIn((head, tail), edges)
