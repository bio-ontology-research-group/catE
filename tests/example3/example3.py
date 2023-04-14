import sys
sys.path.append('..')
from src.cat_projector import CategoricalProjector

from mowl.datasets import PathDataset
from src.edge import Node
from mowl.owlapi import OWLAPIAdapter


a = "http://mowl/A"
c = "http://mowl/C"
d = "http://mowl/D"
r = "http://mowl/R"
ex_r_c = f"{r} some {c}"
all_r_d = f"{r} only {d}"
ex_r_not_d = f"{r} some not {d}"

not_a = f"not {a}"
not_c = f"not {c}"
not_d = f"not {d}"

bot = "owl:Nothing"
top = "owl:Thing"

adapter = OWLAPIAdapter()
owl_a= adapter.create_class(a)
node_a = Node(owl_class = owl_a)


edges = [(a, ex_r_c),
         (a, all_r_d),
         (d, not_c),
         
         (f"{c} and {not_c}", bot),
         (f"{c} and {d}", bot),
         (c, not_d),
         
         (ex_r_c, f"DOMAIN_{r}_under_{ex_r_c}"),
         (f"DOMAIN_{r}_under_{ex_r_c}", ex_r_c),
         (f"CODOMAIN_{r}_under_{ex_r_c}", c),
         
         (all_r_d, f"NOT_DOMAIN_{r}_under_{ex_r_not_d}"),
         (f"NOT_DOMAIN_{r}_under_{ex_r_not_d}", all_r_d),
         (ex_r_not_d, f"DOMAIN_{r}_under_{ex_r_not_d}"),
         (f"DOMAIN_{r}_under_{ex_r_not_d}", ex_r_not_d),
         (f"CODOMAIN_{r}_under_{ex_r_not_d}", not_d),
         
         (f"DOMAIN_{r}_under_{ex_r_c}", f"DOMAIN_{r}_under_{ex_r_not_d}"),
         (a, f"DOMAIN_{r}_under_{ex_r_not_d}"),
         (a, f"NOT_DOMAIN_{r}_under_{ex_r_not_d}"),
         (a, bot),

         (f"DOMAIN_{r}_under_{ex_r_not_d}", not_a),
         (a, not_a),
         (f"{r}_under_{ex_r_c}", r),
]
edges = set(edges)

class Example3():
    def __init__(self):
        pass

    def single_test(self, filename):
        print(f"\n\n")
        print(f"Testing {filename}")
        with open(filename, "r") as f:
            constructed_graph = set()
            for line in f:
                line = line.rstrip("\n")
                src, rel, dst = line.split("\t")
                constructed_graph.add((src, dst))

        found_edges = set()
        not_found_edges = set()
        for edge in edges:
            if edge in constructed_graph:
                found_edges.add(edge)
            else:
                not_found_edges.add(edge)
        print(f"Found edges: {len(found_edges)}")
        for edge in not_found_edges:
            print(edge)
        print(f"Done testing {filename}\n\n")
        
    def test(self):
        ontology = "example3.owl"
        ds = PathDataset(ontology)
        projector = CategoricalProjector()
        projector.project(ds.ontology)
        graph = projector.graph

        print("--------------")
        outfile = ontology.replace(".owl", ".cat.new.edgelist")
        print(f"Graph computed. Writing into file: {outfile}")


        with open(outfile, "w") as f:
             edges = graph.as_str_edgelist()
             for src, rel, dst in edges:
                 f.write(f"{src}\t{rel}\t{dst}\n")
        self.single_test(outfile)

        curr_num_edges = len(graph.as_edgelist())
        for i in range(1, 15):
            print("--------------")
            graph.saturate()
            graph.transitive_closure()
            new_num_edges = len(graph.as_edgelist())
            if new_num_edges == curr_num_edges:
                print("No new edges found. Stopping.")
                break
            curr_num_edges = new_num_edges

            if graph.is_unsatisfiable(node_a):
                print(f"Unsatisfiable: {owl_a}")
                break
            
            outfile = ontology.replace(".owl", f".cat.new.{i}.edgelist")
            print(f"Graph computed. Writing into file: {outfile}")
            with open(outfile, "w") as f:
                edges = graph.as_str_edgelist()
                for src, rel, dst in edges:
                    f.write(f"{src}\t{rel}\t{dst}\n")
            self.single_test(outfile)

           
        
        print("Done.")


if __name__ == "__main__":
    e = Example3()
    e.test()
