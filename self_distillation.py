import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import copy
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric import datasets
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import BatchNorm, MessagePassing

from utils import get_dataset, set_random_seeds

class GNN_SD(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer_norm=False):
        super().__init__()
        self.layer_norm = layer_norm
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            self.layer_norms.append(BatchNorm(hidden_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                if self.layer_norm:
                    x = self.layer_norms[i](x)
                x = x.relu_()
                x = F.dropout(x, p=args.dropout, training=self.training)
        return x
        
def train(epoch, data, data_loader):
    model.train()
    optimizer.zero_grad()
    if epoch > 10:
        y_all = []
        for batch in data_loader:
            y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
            y_all.append(y_hat)
        y_hat = torch.cat(y_all, dim=0)
        edge_index = torch.tensor([[], []], dtype=torch.int64, device=device)
        y_tilde = model(data.x, edge_index)
        
        loss = F.cross_entropy(y_hat[data.train_mask], data.y[data.train_mask])
        loss += (1-args.gamma)*F.cross_entropy(y_tilde[data.train_mask], data.y[data.train_mask])
        loss += args.gamma*KLloss(F.log_softmax(y_tilde, dim=-1),
                                  F.log_softmax(y_hat.detach(), dim=-1))
    else:
        y_all = []
        for batch in data_loader:
            y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
            y_all.append(y_hat)
        y_hat = torch.cat(y_all, dim=0)
        loss = F.cross_entropy(y_hat[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=False)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accs


parser = argparse.ArgumentParser(description='Cora, Citeseer, Pubmed')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
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
edge_index = data.edge_index

if not os.path.exists(f'./param/sd_param/{args.dataset}'):
    os.makedirs(f'./param/sd_param/{args.dataset}')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

kwargs = {'batch_size': 4096, 'num_workers': 1, 'persistent_workers': False}
data_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                             num_neighbors=[5, 5], shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=g, **kwargs)

model = GNN_SD(dataset.num_features, 128, dataset.num_classes, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
KLloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

cnt = 0
patience = 100
best_val_acc = 0
for epoch in tqdm(range(1, 201)):
    loss = train(epoch, data, data_loader)
    if epoch % 1 == 0:
        train_acc, val_acc, test_acc = test(model, data)
        if val_acc > best_val_acc:
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
            print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                 f'Test: {test_acc:.4f}')
            best_val_acc = val_acc
            torch.save({'model': model.state_dict()}, f'./param/sd_param/{args.dataset}/{args.dataset}_distill_model.pt')
            cnt = 0
        else:
            cnt += 1
        if cnt == patience:
            print('early stopping!!!')
            break

checkpoint = torch.load(f'./param/sd_param/{args.dataset}/{args.dataset}_distill_model.pt',
                        map_location=device)
model.load_state_dict(checkpoint['model'], strict=True)
model.to(torch.device('cpu'))
model.eval()

out = model(data.x.cpu(), data.edge_index.cpu())

if not os.path.exists(f'./output/sd/{dataname}'):
    os.makedirs(f'./output/sd/{dataname}')
torch.save(out.detach(), f'./output/sd/{dataname}/'+dataname+f'_{args.seed}.pt')