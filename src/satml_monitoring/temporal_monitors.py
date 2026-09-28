"""Compact faithful GDN and TranAD monitor implementations."""

from __future__ import annotations

import math
import torch
from torch import nn


class GDN(nn.Module):
    """Learned top-k graph-attention next-step forecaster."""

    def __init__(self, sensors: int, window: int, embedding_dim: int = 16,
                 hidden_dim: int = 32, top_k: int = 5):
        super().__init__()
        self.sensors, self.window, self.top_k = sensors, window, top_k
        self.embedding = nn.Parameter(torch.randn(sensors, embedding_dim) * 0.1)
        self.history = nn.Linear(window, hidden_dim)
        self.attention = nn.Linear(2 * (hidden_dim + embedding_dim), 1)
        self.output = nn.Sequential(nn.Linear(hidden_dim + embedding_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def graph(self) -> torch.Tensor:
        normalized = nn.functional.normalize(self.embedding, dim=1)
        similarity = normalized @ normalized.T
        diagonal = torch.eye(self.sensors, dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(diagonal, -torch.inf)
        return torch.topk(similarity, min(self.top_k, self.sensors - 1), dim=1).indices

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        # window: batch, time, sensors
        history = torch.relu(self.history(window.transpose(1, 2)))
        nodes = torch.cat([history, self.embedding.unsqueeze(0).expand(len(window), -1, -1)], dim=2)
        neighbors = self.graph()
        outputs = []
        for sensor in range(self.sensors):
            indices = torch.cat([torch.tensor([sensor], device=window.device), neighbors[sensor]])
            source = nodes[:, sensor:sensor + 1].expand(-1, len(indices), -1)
            candidate = nodes[:, indices]
            weights = torch.softmax(torch.nn.functional.leaky_relu(
                self.attention(torch.cat([source, candidate], dim=2)).squeeze(-1)), dim=1)
            aggregate = torch.sum(weights.unsqueeze(-1) * candidate, dim=1)
            outputs.append(self.output(aggregate).squeeze(-1))
        return torch.stack(outputs, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, length: int):
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(length, dimension)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:values.shape[1]]


class TranAD(nn.Module):
    """Two-phase focus-conditioned transformer reconstructor."""

    def __init__(self, sensors: int, window: int, model_dim: int = 32,
                 heads: int = 4, layers: int = 1):
        super().__init__()
        self.input = nn.Linear(2 * sensors, model_dim)
        self.position = PositionalEncoding(model_dim, window)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim, heads, 2 * model_dim, dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.phase1 = nn.Linear(model_dim, sensors)
        self.phase2 = nn.Linear(model_dim, sensors)

    def reconstruct(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_focus = torch.zeros_like(window)
        hidden1 = self.encoder(self.position(self.input(torch.cat([window, zero_focus], dim=2))))
        first = self.phase1(hidden1)
        focus = (first - window).pow(2)
        hidden2 = self.encoder(self.position(self.input(torch.cat([window, focus], dim=2))))
        second = self.phase2(hidden2)
        return first, second

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(window)[1]
