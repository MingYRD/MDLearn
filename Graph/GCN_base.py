import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as gnn
from torch_geometric.datasets import TUDataset

class GCN_test(torch.nn.Module):

    def __init__(self, features, hidden, classes):
        super(GCN_test, self).__init__()
        self.conv1 = gnn.GCNConv(features, hidden)
        self.conv2 = gnn.GCNConv(hidden, classes)
        