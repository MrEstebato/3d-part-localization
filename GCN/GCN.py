import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch


class AttentionModule(torch.nn.Module):

    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.setup_weights(dim)
        self.init_parameters()

    def setup_weights(self, dim):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(dim, dim))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(
            torch.mm(embedding, transformed_global.view(-1, 1))
        )
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class GCN(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 32
        self.bottle_neck_neurons_1 = 32
        self.bottle_neck_neurons_2 = 8

        self.convolution_1 = GCNConv(
            in_channels=self.number_labels, out_channels=self.filters_1
        )
        self.convolution_2 = GCNConv(
            in_channels=self.filters_1, out_channels=self.filters_2
        )
        self.convolution_3 = GCNConv(
            in_channels=self.filters_2, out_channels=self.filters_3
        )
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(
            self.filters_3, self.bottle_neck_neurons_1
        )
        self.fully_connected_second = nn.Linear(
            self.bottle_neck_neurons_1, self.bottle_neck_neurons_2
        )
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons_2, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(
            features, p=self.dropout, training=self.training
        )
        features = self.convolution_2(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(
            features, p=self.dropout, training=self.training
        )
        features = self.convolution_3(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(
            features, p=self.dropout, training=self.training
        )

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = nn.functional.relu(self.fully_connected_second(scores))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN2(nn.Module):
    def __init__(self, feature_dim_size, num_classes=2, dropout=0.3):
        super(GCN2, self).__init__()

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


class GCN3(nn.Module):
    def __init__(self, feature_dim_size, num_classes=2, dropout=0.3):
        super(GCN3, self).__init__()

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

class GCN4(nn.Module):
    def __init__(self, feature_dim_size, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(feature_dim_size, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)
        self.dropout = dropout

    def forward(self, features, adj, batch):
        x = self.conv1(features, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = geom_nn.global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out(x)
        return x