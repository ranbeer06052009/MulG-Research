# modules.py
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    GRU-based text encoder
    """

    def __init__(self, in_dim, hidden_dim=50):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x : (B, T_l, in_dim)
        """
        out, _ = self.gru(x)
        return out   # (B, T_l, hidden_dim)


class AudioEncoder(nn.Module):
    """
    GRU-based audio encoder
    """

    def __init__(self, in_dim, hidden_dim=50):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x : (B, T_a, in_dim)
        """
        out, _ = self.gru(x)
        return out   # (B, T_a, hidden_dim)


class VisionEncoder(nn.Module):
    """
    GRU-based vision encoder
    """

    def __init__(self, in_dim, hidden_dim=50):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x : (B, T_v, in_dim)
        """
        out, _ = self.gru(x)
        return out   # (B, T_v, hidden_dim)
