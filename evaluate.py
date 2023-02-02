import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch_scatter
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric import datasets
from torch_geometric.utils import to_undirected, degree, add_self_loops
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import BatchNorm, MessagePassing

from utils import get_dataset, set_random_seeds


class LabelProp(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__()
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

from torch_geometric.nn import SAGEConv
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm

class GraphCP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.layer_norms = nn.ModuleList()
        self.layer_norms.append(BatchNorm(hidden_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.layer_norms[i](x)
                x = x.relu_()
                x = F.dropout(x, p=0.0, training=self.training)
        return x
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = self.layer_norms[i](x)
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
    
    def knn_homophily_test(self, data, num_classes, dataset_n, k=9, K=11):
        row, col = data.edge_index.cpu()
        deg = degree(col, data.x.size(0))

        mask = (deg <= K) & data.test_mask & (deg > 0)
        mask_index = mask.nonzero().squeeze()

        if dataset_n == "coauthor-physics":
            y_hit = torch.sigmoid(model.inference(data.x, test_loader).cpu())
        else:
            y_hit = torch.sigmoid(model(data.x, data.edge_index).cpu())
        
        exit_edges_weight = y_hit[col, row]
        # mask self-loop
        y_hit[row, col] = 0
        # mask existing edges
        y_hit[torch.arange(data.x.size(0)), torch.arange(data.x.size(0))] = 0
        y_hit_top = torch.topk(y_hit, k, largest=True)
        y_hit_indices, y_hit_score = y_hit_top.indices, y_hit_top.values
        add_edges_weight = y_hit_score[mask].view(1, -1).squeeze()
        
        # train acc
        row = mask_index.view(-1, 1).repeat(1, k).view(-1)
        col = y_hit_indices[mask].view(-1)
        add_edge_index = torch.stack([col, row], dim=0).cpu()
        
        self_loop = torch.tensor([range(num_nodes), range(num_nodes)], dtype=torch.int64)
        self_loop_weight = torch.ones(self_loop.size(1), dtype=torch.float32)
    
        edges_index = torch.cat([add_edge_index, data.edge_index.cpu(), self_loop], dim=1)
        edges_weights = torch.cat([add_edges_weight.cpu(), exit_edges_weight.cpu(), self_loop_weight.cpu()])

        return edges_index, edges_weights


class Evaluator(torch.nn.Module):
        
    def load_data(self, dataset, seed):
        set_random_seeds(seed)
        data, num_classes, dataset = get_dataset('./data', dataset)
        return data
    
    def degree_group(self, data, _degree, y_pred, threshold):
        data.y = data.y.squeeze()
        num_classes = torch.unique(data.y).size(0)
        max_degree = int(_degree[data.test_mask].max().item())
        micro_results = []
        if threshold > max_degree:
            threshold = max_degree
        for i in range(threshold+1):
            if i == threshold:
                mask = data.test_mask & (_degree.cpu() >= threshold)
            else:
                mask = data.test_mask & (_degree.cpu() == i)
            if mask.sum() == 0:
                micro_results.append(np.nan)
                continue
            micro_f1 = metrics.f1_score(data.y[mask], y_pred[mask], average='micro')
            micro_results.append(micro_f1)
        
        return micro_results
    
    def average_degree_group(self, dataset, path, threshold, model, seed, k, K):
        """evaluate the performance via weighted label propagation
        """
        data = self.load_data(dataset, seed)
        row, col = data.edge_index
        deg = degree(col, data.x.size(0))
        # weighted label propagtaion
        out = torch.load(path+f'./{dataset}_{seed}.pt')
        if k==0 or K==0:
            add_edge_index = add_self_loops(data.edge_index)[0]
            add_edge_weight = torch.ones(add_edge_index.size(1), dtype=torch.float32)
        else:
            add_edge_index, add_edge_weight = model.knn_homophily_test(data, data.y.max().item()+1, dataset,
                                                                       k=k, K=K)
        mp = LabelProp(aggr='add')
        y_pred = mp.propagate(add_edge_index.cpu(), x=F.softmax(out.cpu(), dim=-1),
                              edge_weight=add_edge_weight.cpu())
        y_pred = y_pred.argmax(dim=-1)
        # calculate the acc of degree group
        micro_results = self.degree_group(data, deg, y_pred, threshold)
        micro_f1 = metrics.f1_score(data.y[data.test_mask], y_pred[data.test_mask], 
                                    average='micro')
        # calculate metrics
        f1_score = np.around(np.nanmean(micro_f1), 4)
        group_mean = np.around(np.nanmean(micro_results), 4)
        group_bias = np.around(np.nanstd(micro_results), 4)
        
        return f1_score, group_mean, group_bias


parser = argparse.ArgumentParser(description='Cora, Citeseer, Pubmed')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--k', type=int, default=9)
parser.add_argument('--K', type=int, default=11)
args = parser.parse_args()

# load dataset
set_random_seeds(args.seed)
data, num_classes, dataset = get_dataset('./data', args.dataset)
# load gc model
num_nodes = data.x.size(0)
model = GraphCP(dataset.num_features, 512, num_nodes)
checkpoint = torch.load(f'./param/gc_param/{args.dataset}/{args.dataset}_gc_param_{args.seed}.pt',
                        map_location=data.x.device)
model.load_state_dict(checkpoint['model'], strict=True)

evaluator = Evaluator()
path = f'./output/sd/{args.dataset}/'
if args.dataset == 'cora' or args.dataset == 'citeseer':
    degmax = 15
else:
    degmax = 50
f1, gmean, gbias = evaluator.average_degree_group(args.dataset, path, degmax, model, args.seed, args.k, args.K)
print(f'Micro-F1: {f1}, G.Mean: {gmean}, G.Bias: {gbias}')
