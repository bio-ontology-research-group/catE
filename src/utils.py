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
    "catnit": "http://arrow"
}

bot_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Nothing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Nothing",
    "rdf": "http://www.w3.org/2002/07/owl#Nothing",
    "cat": "owl:Nothing",
    "cat1": "owl:Nothing",
    "cat2": "owl:Nothing",
    "catnit": "owl:Nothing"
    
}

top_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Thing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Thing",
    "rdf": "http://www.w3.org/2002/07/owl#Thing",
    "cat": "owl:Thing",
    "cat1": "owl:Thing",
    "cat2": "owl:Thing",
    "catnit": "owl:Thing"
}



prefix = {
    "pizza": "pizza",
    "dideo": "dideo",
    "fobi": "fobi",
    "go": "go",
    "go_comp": "go.train",
    "foodon_comp": "foodon-merged.train"
}
suffix = {
    "taxonomy": "taxonomy_initial_terminal_no_leakage.edgelist",
    "onto2graph": "onto2graph_initial_terminal_no_leakage.edgelist",
    "owl2vec": "owl2vec_initial_terminal_no_leakage.edgelist",
    "rdf": "rdf_initial_terminal_no_leakage.edgelist",
    "cat": "cat_initial_terminal_no_leakage.edgelist",
    "cat1": "cat.s1_initial_terminal_no_leakage.edgelist",
    "cat2": "cat.s2_initial_terminal_no_leakage.edgelist",
    
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


