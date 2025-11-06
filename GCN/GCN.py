import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GCNConv
import torch


class GCN(nn.Module):
    def __init__(self, feature_dim_size, num_classes=2, dropout=0.3):
        super(GCN, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels=feature_dim_size, out_channels=32)
        self.norm1 = nn.LayerNorm(32)

        self.conv2 = GCNConv(in_channels=32, out_channels=16)
        self.norm2 = nn.LayerNorm(16)

        self.conv3 = GCNConv(in_channels=16, out_channels=16)
        self.norm3 = nn.LayerNorm(16)

        pooled_dim = 2 * 16  # mean + max pooling
        self.fc1 = nn.Linear(pooled_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, adj, features):
        # Layer 1
        h = self.conv1(features, adj)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, adj)
        h = self.norm2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h_in = h
        h = self.conv3(h, adj)
        h = self.norm3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + h_in

        # Global pooling
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        graph_repr = torch.cat([h_mean, h_max], dim=0).unsqueeze(0)

        # Classifier
        out = F.relu(self.fc1(graph_repr))
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.fc2(out)

        # return F.log_softmax(logits, dim=1)
        return logits


class GCN2(nn.Module):
    def __init__(self, feature_dim_size, num_classes=2, dropout=0.3):
        super(GCN2, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels=feature_dim_size, out_channels=64)
        self.norm1 = nn.LayerNorm(64)

        self.conv2 = GCNConv(in_channels=64, out_channels=32)
        self.norm2 = nn.LayerNorm(32)

        self.conv3 = GCNConv(in_channels=32, out_channels=32)
        self.norm3 = nn.LayerNorm(32)

        pooled_dim = 2 * 32  # mean + max pooling
        self.fc1 = nn.Linear(pooled_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, adj, features):
        # Layer 1
        h = self.conv1(features, adj)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, adj)
        h = self.norm2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h_in = h
        h = self.conv3(h, adj)
        h = self.norm3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + h_in

        # Global pooling
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        graph_repr = torch.cat([h_mean, h_max], dim=0).unsqueeze(0)

        # Classifier
        out = F.relu(self.fc1(graph_repr))
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.fc2(out)

        # return F.log_softmax(logits, dim=1)
        return logits


class GCN3(nn.Module):
    def __init__(self, feature_dim_size, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(feature_dim_size, 48)
        self.conv2 = GCNConv(48, 48)
        self.conv3 = GCNConv(48, 48)
        self.fc1 = nn.Linear(96, 48)
        self.out = nn.Linear(48, 2)
        self.dropout = dropout

    def forward(self, features, adj, batch):
        x = self.conv1(features, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = geom_nn.global_mean_pool(x, batch)
        x_max = geom_nn.global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out(x)
        return x
