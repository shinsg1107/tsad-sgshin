import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LearnableAttnPooling(nn.Module):
    """
    score_{i,l} = w^T h_{i,l} + b
    alpha = softmax(score over time)
    c_i = sum_l alpha_{i,l} h_{i,l}

    Input:  h_seq [B, N, T, D] or [B, T, D]
    Output: c     [B, N, D]   or [B, D]
            alpha [B, N, T]   or [B, T]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim) * 0.02)  # [D]
        self.b = nn.Parameter(torch.zeros(()))          # scalar

    def forward(self, h_seq: torch.Tensor):
        squeeze_n = False
        if h_seq.dim() == 3:
            h_seq = h_seq.unsqueeze(1)  # [B,1,T,D]
            squeeze_n = True

        B, N, T, D = h_seq.shape
        assert D == self.dim

        # Flatten (B,N) so we can do a single matmul
        h_bn = rearrange(h_seq, "b n t d -> (b n) t d")        # [B*N, T, D]
        scores = h_bn @ self.w + self.b                       # [B*N, T]
        alpha_bn = F.softmax(scores, dim=-1)                  # [B*N, T]

        # weighted sum: [B*N, D]
        c_bn = (alpha_bn.unsqueeze(-1) * h_bn).sum(dim=1)

        # restore
        c = rearrange(c_bn, "(b n) d -> b n d", b=B, n=N)      # [B,N,D]
        alpha = rearrange(alpha_bn, "(b n) t -> b n t", b=B, n=N)  # [B,N,T]

        if squeeze_n:
            c = c.squeeze(1)       # [B,D]
            alpha = alpha.squeeze(1)  # [B,T]

        return c, alpha
