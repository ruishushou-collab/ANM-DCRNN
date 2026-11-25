import torch
from torch import nn
from torch.nn import functional as F


def row_normalize(matrix):
    return matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-12)


class CompositeGraph(nn.Module):
    def __init__(self, adjacency, embedding_dim=8, alpha=0.5, temperature=1.0,
                 adaptive=True):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        adjacency.fill_diagonal_(0)
        self.register_buffer("forward_support", row_normalize(adjacency))
        self.register_buffer("reverse_support", row_normalize(adjacency.T))
        self.register_buffer("off_diagonal", ~torch.eye(len(adjacency), dtype=torch.bool))
        self.alpha, self.temperature, self.adaptive = alpha, temperature, adaptive
        if adaptive:
            self.source_embedding = nn.Parameter(torch.empty(len(adjacency), embedding_dim))
            self.target_embedding = nn.Parameter(torch.empty(embedding_dim, len(adjacency)))
            nn.init.xavier_uniform_(self.source_embedding)
            nn.init.xavier_uniform_(self.target_embedding)

    def forward(self):
        supports = [self.forward_support, self.reverse_support]
        if self.adaptive:
            scores = F.leaky_relu(self.source_embedding @ self.target_embedding, 0.1)
            weights = F.softmax(scores.masked_fill(~self.off_diagonal, -1e9) / self.temperature, dim=-1)
            weights = row_normalize(weights * self.off_diagonal)
            supports.append(self.alpha * weights)
        return supports


class DiffusionConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_supports, steps=2):
        super().__init__()
        self.steps = steps
        self.linear = nn.Linear(input_dim * (1 + num_supports * steps), output_dim)

    def forward(self, features, supports):
        terms = [features]
        for support in supports:
            current = features
            for _ in range(self.steps):
                current = torch.einsum("ij,bjf->bif", support, current)
                terms.append(current)
        return self.linear(torch.cat(terms, dim=-1))


class DiffusionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_supports, steps=2, dropout=0.0):
        super().__init__()
        self.gates = DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, num_supports, steps)
        self.candidate = DiffusionConv(input_dim + hidden_dim, hidden_dim, num_supports, steps)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, supports):
        reset, update = torch.sigmoid(self.gates(torch.cat([inputs, hidden], -1), supports)).chunk(2, -1)
        candidate = torch.tanh(self.candidate(torch.cat([inputs, reset * hidden], -1), supports))
        return self.dropout(self.norm(update * hidden + (1 - update) * candidate))


class EdgeReadout(nn.Module):
    def __init__(self, hidden_dim, edge_feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, hidden, edges, edge_features):
        static = edge_features.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        combined = torch.cat([hidden[:, edges[:, 0]], hidden[:, edges[:, 1]], static], -1)
        return self.mlp(combined).squeeze(-1)


class GraphRuleNet(nn.Module):
    def __init__(self, adjacency, edges, edge_features, input_dim, forecast_dim,
                 hidden_dim=32, embedding_dim=8, alpha=0.5, adaptive=True):
        super().__init__()
        self.graph = CompositeGraph(adjacency, embedding_dim, alpha, adaptive=adaptive)
        self.num_nodes = self.graph.forward_support.shape[0]
        edges = torch.as_tensor(edges, dtype=torch.long).detach().clone()
        edge_features = torch.as_tensor(edge_features, dtype=torch.float32).detach().clone()
        self.register_buffer("edges", edges)
        self.register_buffer("edge_features", edge_features)
        degree = torch.bincount(edges.flatten(), minlength=self.num_nodes).float().clamp_min(1)
        self.register_buffer("incident_degree", degree[None, :, None])
        self.hidden_dim = hidden_dim
        supports = 3 if adaptive else 2
        self.encoder = DiffusionGRU(input_dim, hidden_dim, supports)
        self.decoder = DiffusionGRU(1 + forecast_dim, hidden_dim, supports)
        self.readout = EdgeReadout(hidden_dim, edge_features.shape[1])
        self.feedback_projection = nn.Linear(1, 1)

    def edge_feedback(self, prediction):
        total = prediction.new_zeros(prediction.shape[0], self.num_nodes, 1)
        for endpoints in self.edges.T:
            total.index_add_(1, endpoints, prediction.unsqueeze(-1))
        return self.feedback_projection(total / self.incident_degree)

    def forward(self, history, forecasts, targets=None, teacher_forcing=0.0):
        supports = self.graph()
        hidden = history.new_zeros(history.shape[0], self.num_nodes, self.hidden_dim)
        for inputs in history.unbind(1):
            hidden = self.encoder(inputs, hidden, supports)
        feedback = history.new_zeros(history.shape[0], self.num_nodes, 1)
        outputs = []
        for lead, known_forecast in enumerate(forecasts.unbind(1)):
            hidden = self.decoder(torch.cat([feedback, known_forecast], -1), hidden, supports)
            prediction = self.readout(hidden, self.edges, self.edge_features)
            outputs.append(prediction)
            previous = prediction
            if self.training and targets is not None and torch.rand((), device=history.device) < teacher_forcing:
                previous = targets[:, lead]
            feedback = self.edge_feedback(previous)
        return torch.stack(outputs, 1)
