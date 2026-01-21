# fusion/mulg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, d_src, d_tgt):
        super().__init__()
        self.W = nn.Linear(d_tgt, d_src, bias=False)

    def forward(self, X_src, X_tgt):
        X_tgt_proj = self.W(X_tgt)
        scores = torch.bmm(X_src, X_tgt_proj.transpose(1, 2))
        alpha = torch.softmax(scores, dim=-1)
        return torch.bmm(alpha, X_tgt)


class CrossModalGRU(nn.Module):
    def __init__(self, d_src, d_tgt, hidden_dim):
        super().__init__()
        self.attn = CrossModalAttention(d_src, d_tgt)
        self.gru = nn.GRU(d_src + d_tgt, hidden_dim, batch_first=True)

    def forward(self, X_src, X_tgt):
        X_tgt_att = self.attn(X_src, X_tgt)
        X = torch.cat([X_src, X_tgt_att], dim=-1)
        out, _ = self.gru(X)
        return out


class MulGFusion(nn.Module):
    """
    MulG fusion module (ASYNC)
    """
    def __init__(self, d_l, d_a, d_v, hidden_dim):
        super().__init__()

        self.l_a = CrossModalGRU(d_l, d_a, hidden_dim)
        self.l_v = CrossModalGRU(d_l, d_v, hidden_dim)
        self.a_l = CrossModalGRU(d_a, d_l, hidden_dim)
        self.a_v = CrossModalGRU(d_a, d_v, hidden_dim)
        self.v_l = CrossModalGRU(d_v, d_l, hidden_dim)
        self.v_a = CrossModalGRU(d_v, d_a, hidden_dim)

    def forward(self, modalities):
        X_v, X_a, X_l = modalities  # IMPORTANT: loader order

        hl = self.l_a(X_l, X_a) + self.l_v(X_l, X_v)
        ha = self.a_l(X_a, X_l) + self.a_v(X_a, X_v)
        hv = self.v_l(X_v, X_l) + self.v_a(X_v, X_a)

        hl = hl[:, -1]
        ha = ha[:, -1]
        hv = hv[:, -1]

        return torch.cat([hl, ha, hv], dim=1)
