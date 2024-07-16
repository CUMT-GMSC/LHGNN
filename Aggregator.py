import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Aggregator(nn.Module):
    def __init__(self, layers, attention_head, output_dim):
        super(Aggregator, self).__init__()
        Layers = []
        for i in range(len(layers) - 1):
            Layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)

        # 基于注意力的聚合器
        node_aggregate_layer = TransformerEncoderLayer(d_model=output_dim, nhead=attention_head, dim_feedforward=output_dim, dropout=0.4, batch_first=True)
        self.node_aggregation = TransformerEncoder(node_aggregate_layer, num_layers=1)

        self.bn = nn.BatchNorm1d(output_dim)

    # 均值聚合
    def mean_aggregate(self, embeddings):
        edge_embeddings = []
        for embedding in embeddings:
            embedding = embedding.mean(dim=0).squeeze()
            edge_embeddings.append(embedding)
        edge_embeddings = torch.stack(edge_embeddings, dim=0)
        return edge_embeddings

    # 二范数均值
    def norm2_aggregate(self, embeddings):
        edge_embeddings = []
        for embedding in embeddings:
            embedding = embedding ** 2
            embedding = embedding.mean(dim=0).squeeze()
            embedding = torch.sqrt(embedding)
            edge_embeddings.append(embedding)
        edge_embeddings = torch.stack(edge_embeddings, dim=0)
        return edge_embeddings

    # 标准差聚合
    def std_aggregate(self, embeddings):
        edge_embeddings = []
        for embedding in embeddings:
            embedding = torch.std(embedding, dim=0, unbiased=False).squeeze()
            edge_embeddings.append(embedding)
        edge_embeddings = torch.stack(edge_embeddings, dim=0)
        return edge_embeddings

    # 最大最小聚合
    def maxmin_aggregate(self, embeddings):
        edge_embedding = []
        for embedding in embeddings:
            max_val, _ = torch.max(embedding, dim=0)
            min_val, _ = torch.min(embedding, dim=0)
            embedding = max_val - min_val
            edge_embedding.append(embedding)
        edge_embeddings = torch.stack(edge_embedding, dim=0)
        return edge_embeddings

    # 最大聚合
    def max_aggregate(self, embeddings):
        edge_embedding = []
        for embedding in embeddings:
            embedding, _ = torch.max(embedding, dim=0)
            edge_embedding.append(embedding)
        edge_embeddings = torch.stack(edge_embedding, dim=0)
        return edge_embeddings

    # 最小聚合
    def min_aggregate(self, embeddings):
        edge_embedding = []
        for embedding in embeddings:
            embedding, _ = torch.min(embedding, dim=0)
            edge_embedding.append(embedding)
        edge_embeddings = torch.stack(edge_embedding, dim=0)
        return edge_embeddings

    # 基于自注意力的聚合
    def attention_aggregate(self, embeddings):
        edge_embedding = []
        for embedding in embeddings:
            embedding = self.node_aggregation(embedding)
            embedding, _ = torch.max(embedding, dim=0)
            edge_embedding.append(embedding)
        edge_embeddings = torch.stack(edge_embedding, dim=0)
        return edge_embeddings

    def classify(self, embedding):
        embedding = torch.linalg.norm(embedding.unsqueeze(0), dim=0)
        return self.cls(embedding)


    def forward(self, embeddings, agg_mode):
        if agg_mode == 'mean':
            edge_embedding = self.mean_aggregate(embeddings)
        elif agg_mode == 'maxmin':
            edge_embedding = self.maxmin_aggregate(embeddings)
        elif agg_mode == 'max':
            edge_embedding = self.max_aggregate(embeddings)
        elif agg_mode == 'min':
            edge_embedding = self.min_aggregate(embeddings)
        elif agg_mode == 'attention':
            edge_embedding = self.attention_aggregate(embeddings)
        elif agg_mode == 'std':
            edge_embedding = self.std_aggregate(embeddings)
        elif agg_mode == 'mean_std':
            mean_embedding = self.mean_aggregate(embeddings)
            std_embedding = self.std_aggregate(embeddings)
            edge_embedding = torch.cat((mean_embedding, std_embedding), dim=1)
        elif agg_mode == 'norm_maxmin':
            norm2_embedding = self.norm2_aggregate(embeddings)
            maxmin_embedding = self.maxmin_aggregate(embeddings)
            edge_embedding = torch.cat((norm2_embedding, maxmin_embedding), dim=1)

        preds = self.classify(edge_embedding)
        return preds