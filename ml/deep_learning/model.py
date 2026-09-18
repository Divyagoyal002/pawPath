"""1D-CNN for canine gait classification on raw windowed IMU signals.

Mirrors the architecture family validated in the arXiv equine-lameness
1D-CNN study and the canine Scientific Reports deep-learning study cited
in docs/PRD.md section 3 — deep learning generalizes better to unseen
individuals than hand-engineered features, at the cost of needing more
training data.

Input shape: (batch, n_channels=6, window_size=120).
"""

import torch
import torch.nn as nn


class GaitCNN(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 3, window_size: int = 120):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_channels) -> (batch, n_channels, window_size)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.classifier(x)


class GaitCNNLSTM(nn.Module):
    """CNN feature extractor + LSTM for temporal context across a window.

    Optional hybrid variant mentioned in docs/PRD.md section 6.3.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 3, hidden_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, seq, features) for LSTM
        _, (h_n, _) = self.lstm(x)
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # concat both directions
        return self.classifier(h_final)
