import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
