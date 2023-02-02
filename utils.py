import random
import os

import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def get_dataset(root, dataname, transform=NormalizeFeatures(), num_train_per_class=20, num_val_per_class=30):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
        'flickr': (datasets.Flickr, None),
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'pubmed': (datasets.Planetoid, 'PubMed'),
        'reddit': (datasets.Reddit2, None),
        'yelp': (datasets.Yelp, None),
    }

    assert dataname in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))
    
    dataset_class, name = pyg_dataset_dict[dataname]
    if name:
        dataset = dataset_class(root, name=name, transform=transform)
        data = dataset[0]
        if name in ['Photo', 'Computers', 'CS', 'physics']:
            data.train_mask = torch.empty(data.x.size(0), dtype=torch.bool)
            data.val_mask = torch.empty(data.x.size(0), dtype=torch.bool)
            data.test_mask = torch.empty(data.x.size(0), dtype=torch.bool)
        data.train_mask.fill_(False)
        data.val_mask.fill_(False)
        data.test_mask.fill_(False)
        for c in range(dataset.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            perm_idx = torch.randperm(idx.size(0))
            train_idx = idx[perm_idx[:num_train_per_class]]
            data.train_mask[train_idx] = True
            val_idx = idx[perm_idx[num_train_per_class:num_train_per_class+num_val_per_class]]
            data.val_mask[val_idx] = True
        data.test_mask = ~(data.train_mask+data.val_mask)
    else:
        dataset = dataset_class(root+'/'+dataname, transform=transform)
        data = dataset[0]
    
    return data, dataset.num_classes, dataset