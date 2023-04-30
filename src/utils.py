import random
import numpy as np
import torch as th
import os

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
    "go": "go",
    "go_comp": "go.train",
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
    "taxonomy": "_initial_terminal_no_leakage.edgelist",
    "onto2graph": "_initial_terminal_no_leakage.edgelist",
    "owl2vec": "_initial_terminal_no_leakage.edgelist",
    "rdf": "_initial_terminal_no_leakage.edgelist",
    "cat": "_initial_terminal_no_leakage.edgelist",
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



    

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. All tensors must have the same size at dimension 0.
        :param batch_size: batch size to load. Defaults to 32.
        :type batch_size: int, optional
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object. Defaults to False.
        :type shuffle: bool, optional
        """

        # Type checking
        if not all(isinstance(t, th.Tensor) for t in tensors):
            raise TypeError("All non-optional parameters must be Tensors")

        if not isinstance(batch_size, int):
            raise TypeError("Optional parameter batch_size must be of type int")

        if not isinstance(shuffle, bool):
            raise TypeError("Optional parameter shuffle must be of type bool")

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


