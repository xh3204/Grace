import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
import copy
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from collections import Counter

import torch_scatter
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric import datasets
from torch_geometric.utils import to_undirected, degree, add_self_loops
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import BatchNorm

from utils import get_dataset, set_random_seeds
EPS = 1e-15


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
                x = F.dropout(x, p=args.dropout, training=self.training)
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

        
def train(epoch, batch):
    model.train()
    optimizer.zero_grad()
    
    y_hat = model(
        batch.x, 
        batch.edge_index
    )[:batch.batch_size]
    
    # mask labels
    y = lbl[batch.n_id[:batch.batch_size]]
    mask = (class_pred[batch.n_id[:batch.batch_size]].matmul(class_pred.t()) >= 0.3) & (y != 1)
    y_hat[mask] = 0.
    
    loss = BCEloss(y_hat, lbl[batch.n_id[:batch.batch_size]])
    
    loss.backward()
    optimizer.step()

    return loss.item()


parser = argparse.ArgumentParser(description='Cora, Citeseer, Pubmed')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--k', type=int, default=9)
parser.add_argument('--K', type=int, default=11)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--logger', default=False, action='store_true')
args = parser.parse_args()

set_random_seeds(args.seed)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

dataname = args.dataset
data, num_classes, dataset = get_dataset('./data', args.dataset,
                                         transform=T.NormalizeFeatures(),
                                         num_train_per_class=20)
data.y = data.y.squeeze()
data.edge_index = to_undirected(data.edge_index).to(device)
data = data.to(device, 'x', 'y')

gnn_pred = torch.load(f'./output/sd/{args.dataset}/{args.dataset}_{args.seed}.pt').to(device)
gnn_pred = F.softmax(gnn_pred, dim=-1)
index = torch.topk(gnn_pred, num_classes-2, largest=False).indices
class_pred = torch.nn.functional.softmax(gnn_pred.scatter(1, index, -torch.inf), dim=-1)

# construct positive labels
num_nodes = data.x.size(0)
lbl = torch.sparse_coo_tensor(add_self_loops(data.edge_index, num_nodes=num_nodes)[0],
                              torch.ones(data.edge_index.size(1)+num_nodes).to(device),
                              [num_nodes, num_nodes])
lbl = lbl.to_dense()

# ensure reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

if args.dataset != "coauthor-physics":
    kwargs = {'batch_size': 4096, 'num_workers': 1, 'persistent_workers': False}
else:
    kwargs = {'batch_size': 512, 'num_workers': 1, 'persistent_workers': False}
data_loader = NeighborLoader(data, input_nodes=None,
                             num_neighbors=[5, 5], shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=g, **kwargs)
data_loader.data.n_id = torch.arange(num_nodes)
if args.dataset == "coauthor-physics":
    kwargs = {'batch_size': 4096, 'num_workers': 1, 'persistent_workers': False}
    test_loader = NeighborLoader(data, input_nodes=None,
                                     num_neighbors=[-1], shuffle=False,
                                     worker_init_fn=seed_worker,
                                     generator=g, **kwargs)
    test_loader.data.n_id = torch.arange(num_nodes)

model = GraphCP(dataset.num_features, 512, num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
BCEloss = nn.BCEWithLogitsLoss()

if not os.path.exists(f'./param/gc_param/{args.dataset}'):
    os.makedirs(f'./param/gc_param/{args.dataset}')
if not os.path.exists(f'./output/gc/{args.dataset}'):
    os.makedirs(f'./output/gc/{args.dataset}')

if args.dataset == "coauthor-physics":
    patience = 20
else:
    patience = 50
best_loss = 1e8
cnt = 0
edge_index = data.edge_index
for epoch in tqdm(range(1, args.epoch+1)):
    tot_loss, count = 0, 0
    for batch in data_loader:
        loss = train(epoch, batch)
        tot_loss += loss
        count += 1
    tot_loss = tot_loss / count
    if epoch % 1 == 0:
        if tot_loss < best_loss:
            if args.logger:
                print(f'Epoch {epoch:02d}, Loss: {tot_loss:.4f}')
            best_loss = tot_loss
            torch.save({'model': model.state_dict()}, f'./param/gc_param/{args.dataset}/{args.dataset}_gc_param_{args.seed}.pt')
            cnt = 0
        else:
            cnt += 1
    if epoch > 100 and cnt > patience:
        print('early stopping')
        break

checkpoint = torch.load(f'./param/gc_param/{args.dataset}/{args.dataset}_gc_param_{args.seed}.pt',
                        map_location=device)
model.load_state_dict(checkpoint['model'], strict=True)
model.to(device)
model.eval()

add_edge_index, add_edge_weight = model.knn_homophily_test(data, num_classes, args.dataset, k=args.k, K=args.K)
torch.save(add_edge_index, f'./output/gc/{args.dataset}/{args.dataset}_add_edge_index_{args.seed}.pt')
torch.save(add_edge_weight, f'./output/gc/{args.dataset}/{args.dataset}_add_edge_weight_{args.seed}.pt')