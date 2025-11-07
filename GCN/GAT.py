import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class GAT(nn.Module):
    def __init__(self, feature_dim_size, dropout=0.3, heads=4):
        super().__init__()

        hidden = 48

        # GATConv: out_channels is per head → final dimension = hidden * heads
        self.conv1 = GATConv(feature_dim_size, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden*heads, hidden, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden*heads, hidden, heads=heads, dropout=dropout)

        # pooling will concat mean + max → 2 * hidden*heads
        pooled_dim = (hidden*heads) * 2

        self.fc1 = nn.Linear(pooled_dim, 48)
        self.out = nn.Linear(48, 2)

        self.dropout = dropout

    def forward(self, features, adj, batch):
        x = self.conv1(features, adj)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, adj)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out(x)
        return x
