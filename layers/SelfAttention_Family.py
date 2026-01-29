import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from dataclasses import dataclass
from typing import Optional, Tuple

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

class OracleAD_MHSA(nn.Module):
    """
    OracleAD MHSA block with cfg-driven hyperparameters and a switchable backend:

    - BACKEND="torch": uses nn.MultiheadAttention (fast, standard implementation)
    - BACKEND="paper": implements Eq.(8) more literally with head-specific Wq/Wk/Wv/Wo and sum over heads

    Input:
        C: [B, N, D]
    Output:
        C_out: [B, N, D]
        attn:  [B, H, N, N] (head-wise attention if available; otherwise best-effort)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_m = cfg.ORACLEAD.MHSA

        # ---- Required cfg fields (recommended) ----
        self.backend: str = getattr(self.cfg_m, "BACKEND", "torch")  # "torch" or "paper"
        self.dim: int = int(self.cfg_m.DIM)
        self.num_heads: int = int(self.cfg_m.NUM_HEADS)
        self.bias: bool = bool(getattr(self.cfg_m, "BIAS", True))

        # Dropouts
        self.attn_dropout: float = float(getattr(self.cfg_m, "ATTN_DROPOUT", 0.0))
        self.proj_dropout: float = float(getattr(self.cfg_m, "PROJ_DROPOUT", float(getattr(self.cfg_m, "DROPOUT", 0.0))))

        # Block options
        self.pre_norm: bool = bool(getattr(self.cfg_m, "PRE_NORM", True))
        self.residual: bool = bool(getattr(self.cfg_m, "RESIDUAL", True))
        self.mask_self: bool = bool(getattr(self.cfg_m, "MASK_SELF", False))  # optional: block diagonal self-attn

        assert self.dim % self.num_heads == 0, f"DIM({self.dim}) must be divisible by NUM_HEADS({self.num_heads})"
        self.dh = self.dim // self.num_heads

        self.norm = nn.LayerNorm(self.dim)
        self.out_drop = nn.Dropout(self.proj_dropout)

        # cache for attention mask (N may be fixed)
        self._cached_N: Optional[int] = None
        self._cached_mask: Optional[torch.Tensor] = None

        if self.backend == "torch":
            self.mha = nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.attn_dropout,
                bias=self.bias,
                batch_first=True,  # [B, N, D]
            )
        elif self.backend == "paper":
            # Head-specific projections: R^D -> R^{d_h}
            self.Wq = nn.ModuleList([nn.Linear(self.dim, self.dh, bias=self.bias) for _ in range(self.num_heads)])
            self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dh, bias=self.bias) for _ in range(self.num_heads)])
            self.Wv = nn.ModuleList([nn.Linear(self.dim, self.dh, bias=self.bias) for _ in range(self.num_heads)])
            # Head-specific output projection: R^{d_h} -> R^D (and we sum over heads)
            self.Wo = nn.ModuleList([nn.Linear(self.dh, self.dim, bias=self.bias) for _ in range(self.num_heads)])
        else:
            raise ValueError(f"Unknown MHSA BACKEND: {self.backend}. Use 'torch' or 'paper'.")

    def _get_attn_mask(self, N: int, device, dtype) -> torch.Tensor:
        """
        If mask_self=True, create an [N,N] additive mask that blocks diagonal (self-attention).
        PyTorch MultiheadAttention expects float mask with -inf where blocked.
        """
        if (self._cached_mask is None) or (self._cached_N != N) or (self._cached_mask.device != device):
            m = torch.zeros((N, N), device=device, dtype=dtype)
            diag = torch.eye(N, device=device, dtype=torch.bool)
            m[diag] = float("-inf")
            self._cached_mask = m
            self._cached_N = N
        return self._cached_mask

    def forward(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        C: [B, N, D]
        returns:
            C_out: [B, N, D]
            attn:  [B, H, N, N]
        """
        B, N, D = C.shape
        assert D == self.dim, f"Expected input dim={self.dim}, got {D}"

        x = self.norm(C) if self.pre_norm else C

        if self.backend == "torch":
            attn_mask = None
            if self.mask_self:
                attn_mask = self._get_attn_mask(N, device=C.device, dtype=C.dtype)

            # average_attn_weights=False -> head-wise weights (if supported by your torch version)
            attn_out, attn_w = self.mha(
                x, x, x,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            # attn_w is typically [B, H, N, N] (or sometimes [B, N, N] on older versions)
            if attn_w.dim() == 3:
                attn_w = attn_w.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # best-effort
            out = attn_out

        else:  # "paper"
            outs = []
            attns = []
            scale = 1.0 / math.sqrt(self.dh)

            for h in range(self.num_heads):
                Q = self.Wq[h](x)  # [B, N, dh]
                K = self.Wk[h](x)  # [B, N, dh]
                V = self.Wv[h](x)  # [B, N, dh]

                A = torch.matmul(Q, K.transpose(-1, -2)) * scale  # [B, N, N]

                if self.mask_self:
                    A = A + self._get_attn_mask(N, device=C.device, dtype=C.dtype).unsqueeze(0)

                A = F.softmax(A, dim=-1)  # [B, N, N]

                # optional dropout on attention weights (common in transformers)
                if self.attn_dropout > 0:
                    A = F.dropout(A, p=self.attn_dropout, training=self.training)

                O = torch.matmul(A, V)  # [B, N, dh]
                outs.append(self.Wo[h](O))            # [B, N, D]
                attns.append(A.unsqueeze(1))          # [B, 1, N, N]

            out = torch.stack(outs, dim=0).sum(dim=0)  # sum over heads: [B, N, D]
            attn_w = torch.cat(attns, dim=1)           # [B, H, N, N]

        # residual + dropout
        out = self.out_drop(out)
        if self.residual:
            out = C + out

        if not self.pre_norm:
            out = self.norm(out)

        return out, attn_w
