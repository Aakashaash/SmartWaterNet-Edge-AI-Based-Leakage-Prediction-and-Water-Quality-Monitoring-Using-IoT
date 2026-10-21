
import torch
import torch.nn as nn


class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x.unsqueeze(-1)).squeeze(-1)
        return self.dropout(h * self.attn(h))


class SmartWaterNetQuality(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        self.b1 = SpatialTemporalBlock(in_dim, hidden)
        self.b2 = SpatialTemporalBlock(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.b2(self.b1(x)))


class WaterQualityPredictor(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class LeakRateEstimator(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)