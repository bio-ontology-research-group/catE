import torch.nn as nn
import os
import pandas as pd
from pykeen.models import TransE, DistMult, ConvKB, ERModel, RotatE
from mowl.owlapi.defaults import TOP, BOT
import logging
import torch as th
from mowl.utils.data import FastTensorDataLoader
from src.utils import bot_name, top_name, prefix, suffix, suffix_unsat
from pykeen.triples import TriplesFactory

class OrderE(TransE):
    def __init__(self, *args, **kwargs):
        super(OrderE, self).__init__(*args, **kwargs)

    def forward(self, h_indices, r_indices, t_indices, mode = None):
        h, _, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        order_loss = th.linalg.norm(th.relu(t-h), dim=1)
        return -order_loss

    def score_hrt(self, hrt_batch, mode = None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r = hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return -th.linalg.norm(th.relu(t-h), dim=1)

    def distance(self, hrt_batch, mode = None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r = hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        distance = th.linalg.norm(t-h, dim=1)
        return  -distance
    

class Model():
    def __init__(self,
                 use_case,
                 graph_type,
                 kge_model,
                 root,
                 emb_dim,
                 margin,
                 weight_decay,
                 batch_size,
                 lr,
                 num_negs,
                 test_batch_size,
                 epochs,
                 validation_file,
                 test_file,
                 device,
                 seed,
                 initial_tolerance,
                 test_unsatisfiability=False,
                 test_deductive_inference=False,
                 test_completion=False,
                 test_named_classes=False,
                 reduced_subsumption=False,
                 test_existential=False,
                 test_ppi=False
                 ):


        if test_deductive_inference:
            if test_named_classes and test_existential:
                raise ValueError("Cannot test both named classes and existential reasoning")
            if not test_named_classes and not test_existential:
                raise ValueError("Must test either named classes or existential reasoning")
        
        self.use_case = use_case
        self.graph_type = graph_type
        self.kge_model = kge_model
        self.root = root
        self.emb_dim = emb_dim
        self.margin = margin
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.num_negs = num_negs
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.validation_file = validation_file
        self.test_file = test_file
        self.device = device
        self.seed = seed
        self.initial_tolerance = initial_tolerance
        self.test_unsatisfiability = test_unsatisfiability
        self.test_deductive_inference = test_deductive_inference
        self.test_completion = test_completion
        self.test_named_classes = test_named_classes
        self.reduced_subsumption = reduced_subsumption
        self.test_existential = test_existential
        self.test_ppi = test_ppi

        self._triples_factory = None
        self._graph = None
        self._graph_path = None
        self._model_path = None
        self._node_to_id = None
        self._relation_to_id = None
        self._id_to_node = None
        self._id_to_relation = None
        self._ontology_classes = None
        self._ontology_properties = None
        self._ontology_classes_idxs = None
        self._ontology_properties_idxs = None

        self.able_to_validate = False
        if validation_file is not None:
            self.able_to_validate = True

        
        print("Parameters:")
        print(f"\tUse case: {self.use_case}")
        print(f"\tGraph type: {self.graph_type}")
        print(f"\tKGE model: {self.kge_model}")
        print(f"\tRoot: {self.root}")
        print(f"\tEmbedding dimension: {self.emb_dim}")
        print(f"\tMargin: {self.margin}")
        print(f"\tWeight decay: {self.weight_decay}")
        print(f"\tBatch size: {self.batch_size}")
        print(f"\tLearning rate: {self.lr}")
        print(f"\tNumber of negatives: {self.num_negs}")
        print(f"\tTest batch size: {self.test_batch_size}")
        print(f"\tEpochs: {self.epochs}")
        print(f"\tTest file: {self.test_file}")
        print(f"\tDevice: {self.device}")
        print(f"\tSeed: {self.seed}")
        
        self.model = KGEModule(kge_model,
                               triples_factory=self.triples_factory,
                               embedding_dim=self.emb_dim,
                               random_seed=self.seed)

        assert os.path.exists(self.root), f"Root directory {self.root} does not exist."
        assert os.path.exists(self.test_file), f"Test file {self.test_file} does not exist."



    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        
        print(f"Loading graph from {self.graph_path}")
        graph = pd.read_csv(self.graph_path, sep="\t", header=None)
        graph.columns = ["head", "relation", "tail"]
        self._graph = graph
        print("Done")
        return self._graph

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        use_case = prefix[self.use_case]
        graph_name = graph_type[self.graph_type]
        if self.test_unsatisfiability:
            actual_suffix = suffix_unsat[self.graph_type]
        else:
            actual_suffix = suffix[self.graph_type]
        graph_path = os.path.join(self.root, f"{use_case}.{graph_name}{actual_suffix}")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        return self._graph_path

    @property
    def classes_path(self):
        path = os.path.join(self.root, "classes.txt")
        assert os.path.exists(path), f"Classes file {path} does not exist"
        return path

    @property
    def properties_path(self):
        path = os.path.join(self.root, "properties.txt")
        assert os.path.exists(path), f"Properties file {path} does not exist"
        return path

    @property
    def model_path(self):
        if self._model_path is not None:
            return self._model_path

        params_str = f"{self.graph_type}"
        params_str += f"_kge_{self.kge_model}"
        params_str += f"_dim{self.emb_dim}"
        params_str += f"_marg{self.margin}"
        params_str += f"_reg{self.weight_decay}"
        params_str += f"_bs{self.batch_size}"
        params_str += f"_lr{self.lr}"
        params_str += f"_negs{self.num_negs}"
        
        models_dir = os.path.dirname(self.root)
        models_dir = os.path.join(models_dir, "models")

        basename = f"{params_str}.model.pt"
        self._model_path = os.path.join(models_dir, basename)
        return self._model_path

    @property
    def test_tuples_path(self):
        path = self.test_file
        assert os.path.exists(path), f"Test tuples file {path} does not exist"
        return path


    @property
    def validation_tuples_path(self):
        path = self.validation_file
        assert os.path.exists(path), f"Validation tuples file {path} does not exist"
        return path

    

    @property
    def node_to_id(self):
        if self._node_to_id is not None:
            return self._node_to_id

        graph_classes = set(self.graph["head"].unique()) | set(self.graph["tail"].unique())
        bot = bot_name[self.graph_type]
        top = top_name[self.graph_type]
        graph_classes.add(bot)
        graph_classes.add(top)
                
        ont_classes = set(self.ontology_classes)
        all_classes = list(graph_classes | ont_classes)
        all_classes.sort()
        self._node_to_id = {c: i for i, c in enumerate(all_classes)}
        logging.info(f"Number of graph nodes: {len(self._node_to_id)}")
        return self._node_to_id
    
    @property
    def id_to_node(self):
        if self._id_to_node is not None:
            return self._id_to_node
        
        id_to_node =  {v: k for k, v in self.node_to_id.items()}
        self._id_to_node = id_to_node
        return self._id_to_node
    
    @property
    def relation_to_id(self):
        if self._relation_to_id is not None:
            return self._relation_to_id

        graph_rels = list(self.graph["relation"].unique())
        graph_rels.sort()
        self._relation_to_id = {r: i for i, r in enumerate(graph_rels)}
        logging.info(f"Number of graph relations: {len(self._relation_to_id)}")
        return self._relation_to_id

    @property
    def id_to_relation(self):
        if self._id_to_relation is not None:
            return self._id_to_relation

        id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self._id_to_relation = id_to_relation
        return self._id_to_relation

    
    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        tensor = []
        for row in self.graph.itertuples():
            tensor.append([self.node_to_id[row.head],
                           self.relation_to_id[row.relation],
                           self.node_to_id[row.tail]])

        tensor = th.LongTensor(tensor)
        self._triples_factory = TriplesFactory(tensor, self.node_to_id, self.relation_to_id, create_inverse_triples=True)
        return self._triples_factory


    @property
    def ontology_classes(self):
        if self._ontology_classes is not None:
            return self._ontology_classes
        
        classes = pd.read_csv(self.classes_path, sep=",", header=None)
        classes.columns = ["classes"]
        classes = classes["classes"].values.tolist()
        classes = set(classes)
        bot = bot_name[self.graph_type]
        top = top_name[self.graph_type]
        classes.add(bot)
        classes.add(top)
        classes = list(classes)
        classes.sort()

        self._ontology_classes = classes
        return self._ontology_classes
    
    @property
    def ontology_classes_idxs(self):
        if self._ontology_classes_idxs is not None:
            return self._ontology_classes_idxs
        
        class_to_id = {c: self.node_to_id[c] for c in self.ontology_classes}
        ontology_classes_idxs = th.tensor(list(class_to_id.values()), dtype=th.long, device=self.device)
        self._ontology_classes_idxs = ontology_classes_idxs
        return self._ontology_classes_idxs

    @property
    def bot(self):
        return bot_name[self.graph_type]

    @property
    def top(self):
        return top_name[self.graph_type]

    
    @property
    def bot_idx(self):
        return self.ontology_classes.index(self.bot)
    
    @property
    def top_idx(self):
        return self.ontology_classes.index(self.top)
    
    
    
    @property
    def ontology_properties(self):
        if self._ontology_properties is not None:
            return self._ontology_properties
        
        properties = pd.read_csv(self.properties_path, sep=",", header=None)
        properties.columns = ["properties"]
        properties = properties["properties"].values.tolist()

        if self.graph_type in [ "rdf", "cat", "cat1", "cat2"]:
            properties = [r for r in properties if r in self.node_to_id]
        else:
            properties = [r for r in properties if r in self.relation_to_id]

        properties.sort()

        self._ontology_properties = properties
        return self._ontology_properties
    
    @property
    def ontology_properties_idxs(self):
        if self._ontology_properties_idxs is not None:
            return self._ontology_properties_idxs
        
        if self.graph_type in [ "rdf", "cat", "cat1", "cat2"]:
            prop_to_id = {c: self.node_to_id[c] for c in self.ontology_properties if c in self.node_to_id}
        else:
            prop_to_id = {c: self.relation_to_id[c] for c in self.ontology_properties if c in self.relation_to_id}
        
        logging.info(f"Number of ontology properties found in projection: {len(prop_to_id)}")
        ontology_properties_idxs = th.tensor(list(prop_to_id.values()), dtype=th.long, device=self.device)
        
        self._ontology_properties_idxs = ontology_properties_idxs
        return self._ontology_properties_idxs

        
        
    def create_graph_train_dataloader(self):
        heads = [self.node_to_id[h] for h in self.graph["head"]]
        rels = [self.relation_to_id[r] for r in self.graph["relation"]]
        tails = [self.node_to_id[t] for t in self.graph["tail"]]

        heads = th.LongTensor(heads)
        rels = th.LongTensor(rels)
        tails = th.LongTensor(tails)
        
        dataloader = FastTensorDataLoader(heads, rels, tails,
                                          batch_size=self.batch_size, shuffle=True)
        return dataloader
    

    def create_subsumption_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep=",", header=None)
        num_cols = tuples.shape[1]

                
        if num_cols == 2:
            tuples.columns = ["head", "tail"]
        elif num_cols == 3:
            tuples.columns = ["head", "relation", "tail"]
        else:
            raise ValueError(f"Invalid number of columns in {tuples_path}")

        tuples = tuples.drop_duplicates()
        tuples["head"] = tuples["head"].apply(lambda x: bot_name[self.graph_type] if x == "owl:Nothing" else x)
        tuples["head"] = tuples["head"].apply(lambda x: top_name[self.graph_type] if x == "owl:Thing" else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: bot_name[self.graph_type] if x == "owl:Nothing" else x)
        tuples["tail"] = tuples["tail"].apply(lambda x: top_name[self.graph_type] if x == "owl:Thing" else x)
        

        
        heads = [self.node_to_id[h] for h in tuples["head"]]
        tails = [self.node_to_id[t] for t in tuples["tail"]]

        

        heads = th.tensor(heads, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)
        
        if num_cols == 2:
            rel_idx = self.relation_to_id['http://arrow']
            rels = rel_idx * th.ones_like(heads)
        else:
            
            if (self.graph_type in ["rdf", "cat", "cat1", "cat2"]) and self.test_unsatisfiability:
                rels = [self.node_to_id[r] for r in tuples["relation"]]
            else:
                rels = [self.relation_to_id[r] for r in tuples["relation"]]
            rels = th.tensor(rels, dtype=th.long)
        
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

    
    
    def train(self):
        raise NotImplementedError

    def get_filtering_labels(self):
        raise NotImplementedError

    def compute_ranking_metrics(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
