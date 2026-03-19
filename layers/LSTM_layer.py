import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from utils.misc import load_causal_graph

# class LSTMEncoder(nn.Module):
#     def __init__(self,cfg):
#         super(LSTMEncoder, self).__init__()
#         self.cfg = cfg
#         self.cfg_model = cfg.ORACLEAD.LSTM_ENCODER

#         self.n_vars = cfg.DATA.N_VAR
#         self.input_dim = self.cfg_model.INPUT_DIM
#         self.hidden_dim = self.cfg_model.HIDDEN_DIM
#         self.num_layers = self.cfg_model.NUM_LAYERS
#         self.dropout = self.cfg_model.DROPOUT

#         lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
#         self.out_dim = self.hidden_dim

#         def make_lstm():
#             return nn.LSTM(
#                 input_size=self.input_dim,
#                 hidden_size=self.hidden_dim,
#                 num_layers=self.num_layers,
#                 dropout=lstm_dropout,
#                 bidirectional=self.bidirectional,
#                 batch_first=True,  # [B, T, C]
#             )
#         self.lstm_list = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: [B, N, T]  (or [B, N, T, C] if input_dim>1)
#         """
#         if x.dim() == 3:
#             # [B, N, T] -> [B, N, T, 1]
#             x = x.unsqueeze(-1)

#         B, N, T, C = x.shape
#         assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"
#         assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got {C}"
#         # Per-variable LSTM
#         h_seq_list = []
#         h_last_list = []
#         for i in range(N):
#             xi = x[:, i, :, :]              # [B, T, C]
#             h_seq_i, (h_n, c_n) = self.lstm_list[i](xi)  # [B, T, out_dim]
#             if self.bidirectional:
#                 h_last_i = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2*hidden_dim]
#             else:
#                 h_last_i = h_n[-1]  # [B, hidden_dim]
#             h_seq_list.append(h_seq_i.unsqueeze(1))   # [B, 1, T, out_dim]
#             h_last_list.append(h_last_i.unsqueeze(1)) # [B, 1, out_dim]
#         h_seq = torch.cat(h_seq_list, dim=1)   # [B, N, T, out_dim]
#         h_last = torch.cat(h_last_list, dim=1) # [B, N, out_dim]

#         return h_seq, h_last


class LSTMDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_dec = cfg.ORACLEAD.DECODER

        self.n_vars    = int(cfg.DATA.N_VAR)
        self.dim       = int(self.cfg_dec.DIM)
        self.hidden_dim= int(self.cfg_dec.HIDDEN_DIM)
        self.num_layers= int(getattr(self.cfg_dec, "NUM_LAYERS", 1))
        self.dropout   = float(getattr(self.cfg_dec, "DROPOUT", 0.0))
        self.bias      = bool(getattr(self.cfg_dec, "BIAS", True))
        self.out_dim   = int(getattr(self.cfg_dec, "OUT_DIM", 1))
        self.T         = int(getattr(self.cfg_dec, "PAST_LEN", cfg.DATA.WIN_SIZE - 1))
        self.L         = self.T + 1  # full window length

        def make_lstm():
            return nn.LSTM(
                input_size=self.out_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bias=self.bias,
            )

        self.lstm_list   = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])

        # 학습 가능한 hidden/cell 초기화 projection
        self.init_h_list = nn.ModuleList([
            nn.Linear(self.dim, self.num_layers * self.hidden_dim)
            for _ in range(self.n_vars)
        ])
        self.init_c_list = nn.ModuleList([
            nn.Linear(self.dim, self.num_layers * self.hidden_dim)
            for _ in range(self.n_vars)
        ])

        self.out_list = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.out_dim)
            for _ in range(self.n_vars)
        ])

    def forward(self, c_star: torch.Tensor):
        B, N, D = c_star.shape
        assert N == self.n_vars
        assert D == self.dim

        x_hat_past_list = []
        x_hat_next_list = []

        for i in range(N):
            ci = c_star[:, i, :]  # [B, D]
            z = torch.zeros(B, self.L, self.out_dim, device=ci.device, dtype=ci.dtype)

            # [B, num_layers * hidden_dim] → [B, num_layers, hidden_dim] → [num_layers, B, hidden_dim]
            h0 = torch.tanh(self.init_h_list[i](ci))
            h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

            c0 = torch.tanh(self.init_c_list[i](ci))
            c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

            Y, _ = self.lstm_list[i](z, (h0, c0))  # [B, L, hidden_dim]

            O = self.out_list[i](Y).squeeze(-1)     # [B, L]

            recon_i = O[:, :self.T]                 # [B, T]
            next_i  = O[:, self.T]                  # [B]

            x_hat_past_list.append(recon_i.unsqueeze(1))             # [B, 1, T]
            x_hat_next_list.append(next_i.unsqueeze(1).unsqueeze(-1))  # [B, 1, 1]

        x_hat_past = torch.cat(x_hat_past_list, dim=1).unsqueeze(-1)  # [B, N, T, 1]
        x_hat_next = torch.cat(x_hat_next_list, dim=1)                # [B, N, 1]
        
        return x_hat_past, x_hat_next


class LSTMEncoder(nn.Module):
    """
    Per-variable LSTM Encoder (SHARED 제거)

    causal_mode:
      None       : 기존 방식 (input_dim=1)
      'max_pad'  : 방법 2 - max parents padding
      'topk'     : 방법 3 - top-k parents fixed
      'weighted' : 방법 4 - weighted sum (input_dim=1 유지)
      'linear'   : 방법 5 - Linear projection → fixed dim → LSTM
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg       = cfg
        self.cfg_model = cfg.ORACLEAD.LSTM_ENCODER

        self.n_vars       = cfg.DATA.N_VAR
        self.hidden_dim   = self.cfg_model.HIDDEN_DIM
        self.num_layers   = self.cfg_model.NUM_LAYERS
        self.dropout      = self.cfg_model.DROPOUT
        self.bidirectional= self.cfg_model.BIDIRECTIONAL

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        # ------------------------------------------------------------------ #
        # Causal graph 설정
        # ------------------------------------------------------------------ #
        self.causal_mode    : Optional[str]        = None
        self.causal_alpha   : float                = 0.0
        self.topk           : int                  = 3
        self.parent_indices : Optional[List[List[int]]] = None
        self.proj_list      : Optional[nn.ModuleList]   = None
        self.register_buffer("causal_weights", None)

        cfg_causal = getattr(cfg.ORACLEAD, "CAUSAL_ENCODER", None)
        causal_enable = (cfg_causal is not None and
                         bool(getattr(cfg_causal, "ENABLE", False)))

        if causal_enable:
            path = getattr(cfg_causal, "GRAPH_PATH", "")
            if path and os.path.isfile(path):
                cg = load_causal_graph(
                    path,
                    normalize=bool(getattr(cfg_causal, "NORMALIZE", True))
                )
                assert cg.shape == (self.n_vars, self.n_vars), \
                    f"causal graph shape {cg.shape} != ({self.n_vars}, {self.n_vars})"

                self.causal_mode  = getattr(cfg_causal, "MODE", "linear")
                self.causal_alpha = float(getattr(cfg_causal, "ALPHA", 0.3))
                threshold         = float(getattr(cfg_causal, "THRESHOLD", 0.1))
                self.topk         = int(getattr(cfg_causal, "TOPK", 3))
                d_proj            = int(getattr(cfg_causal, "PROJ_DIM", 16))

                # causal weight matrix [N, N]
                self.register_buffer("causal_weights", torch.from_numpy(cg))

                # 변수별 parent 파악
                # cg[i,j] = i→j, 변수 j의 parents = column j
                self.parent_indices = []
                for j in range(self.n_vars):
                    col = cg[:, j]
                    if self.causal_mode == 'topk':
                        k = min(self.topk, self.n_vars - 1)
                        parents = np.argsort(col)[::-1][:k+1].tolist()
                        parents = [p for p in parents if p != j][:k]
                    else:
                        parents = [i for i in range(self.n_vars)
                                   if i != j and col[i] > threshold]
                    self.parent_indices.append(parents)

                # input_size 결정
                max_p = max(len(p) for p in self.parent_indices)

                if self.causal_mode == 'max_pad':
                    input_size = 1 + max_p
                elif self.causal_mode == 'topk':
                    input_size = 1 + self.topk
                elif self.causal_mode == 'weighted':
                    input_size = 1
                elif self.causal_mode == 'linear':
                    # 변수별 Linear projection
                    self.proj_list = nn.ModuleList([
                        nn.Linear(1 + len(self.parent_indices[j]), d_proj)
                        for j in range(self.n_vars)
                    ])
                    input_size = d_proj
                else:
                    input_size = self.cfg_model.INPUT_DIM

                print(f"[LSTMEncoder] causal_mode={self.causal_mode}, "
                      f"alpha={self.causal_alpha}, max_parents={max_p}")
                for j in range(self.n_vars):
                    print(f"  var {j:2d}: {len(self.parent_indices[j])} parents "
                          f"{self.parent_indices[j]}")
            else:
                print(f"[LSTMEncoder] CAUSAL path not found: {path}")
                input_size = self.cfg_model.INPUT_DIM
        else:
            input_size = self.cfg_model.INPUT_DIM

        self.input_dim = input_size

        # ------------------------------------------------------------------ #
        # Per-variable LSTM
        # ------------------------------------------------------------------ #
        def make_lstm(in_size: int) -> nn.LSTM:
            return nn.LSTM(
                input_size=in_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.lstm_list = nn.ModuleList(
            [make_lstm(self.input_dim) for _ in range(self.n_vars)]
        )

    # ------------------------------------------------------------------ #
    # 방법별 입력 구성
    # ------------------------------------------------------------------ #
    def _build_input_max_pad(
        self, x: torch.Tensor, x_lag: torch.Tensor, j: int
    ) -> torch.Tensor:
        """[x_j, p1_{t-1}, ..., 0(pad)] → [B, T, 1+max_p]"""
        B, N, T, _ = x.shape
        max_p   = max(len(p) for p in self.parent_indices)
        parents = self.parent_indices[j]

        parts = [x[:, j, :, :]]  # [B, T, 1]
        for p in parents:
            parts.append(x_lag[:, p, :, :])  # [B, T, 1]

        n_pad = max_p - len(parents)
        if n_pad > 0:
            parts.append(torch.zeros(B, T, n_pad, device=x.device, dtype=x.dtype))

        return torch.cat(parts, dim=-1)  # [B, T, 1+max_p]

    def _build_input_topk(
        self, x: torch.Tensor, x_lag: torch.Tensor, j: int
    ) -> torch.Tensor:
        """[x_j, top1_{t-1}, ..., topk_{t-1}] → [B, T, 1+topk]"""
        B, N, T, _ = x.shape
        parents = self.parent_indices[j]  # 이미 top-k 선택됨

        parts = [x[:, j, :, :]]  # [B, T, 1]
        for p in parents:
            parts.append(x_lag[:, p, :, :])

        n_pad = self.topk - len(parents)
        if n_pad > 0:
            parts.append(torch.zeros(B, T, n_pad, device=x.device, dtype=x.dtype))

        return torch.cat(parts, dim=-1)  # [B, T, 1+topk]

    def _build_input_weighted(
        self, x: torch.Tensor, x_lag: torch.Tensor, j: int
    ) -> torch.Tensor:
        """x_j + alpha * Σ_i cg[i,j] * x_i_{t-1} → [B, T, 1]"""
        x_j      = x[:, j, :, :]                   # [B, T, 1]
        weights  = self.causal_weights[:, j]         # [N]
        x_lag_sq = x_lag.squeeze(-1)                # [B, N, T]
        parent_sum = torch.einsum('n,bnt->bt', weights, x_lag_sq).unsqueeze(-1)
        return x_j + self.causal_alpha * parent_sum  # [B, T, 1]

    def _build_input_linear(
        self, x: torch.Tensor, x_lag: torch.Tensor, j: int
    ) -> torch.Tensor:
        """
        [x_j, p1_{t-1}, p2_{t-1}, ...] → Linear → [B, T, d_proj]
        parent 없으면 Linear(1, d_proj)
        """
        parents = self.parent_indices[j]

        parts = [x[:, j, :, :]]  # [B, T, 1]
        for p in parents:
            parts.append(x_lag[:, p, :, :])  # [B, T, 1]

        x_cat = torch.cat(parts, dim=-1)          # [B, T, 1+num_parents]
        return self.proj_list[j](x_cat)            # [B, T, d_proj]

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        x_lag: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:     [B, N, T] or [B, N, T, 1]
        x_lag: [B, N, T, 1]  1-step lagged (causal 사용 시 자동 생성)
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        B, N, T, C = x.shape
        assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"

        # 1-step lag 자동 생성
        if self.causal_mode is not None and x_lag is None:
            x_lag = torch.zeros_like(x)
            x_lag[:, :, 1:, :] = x[:, :, :-1, :]

        h_seq_list  = []
        h_last_list = []

        for j in range(N):
            # 방법별 입력 구성
            if self.causal_mode == 'max_pad':
                x_j_input = self._build_input_max_pad(x, x_lag, j)
            elif self.causal_mode == 'topk':
                x_j_input = self._build_input_topk(x, x_lag, j)
            elif self.causal_mode == 'weighted':
                x_j_input = self._build_input_weighted(x, x_lag, j)
            elif self.causal_mode == 'linear':
                x_j_input = self._build_input_linear(x, x_lag, j)
            else:
                x_j_input = x[:, j, :, :]  # [B, T, 1] 기존

            h_seq_j, (h_n, _) = self.lstm_list[j](x_j_input)

            if self.bidirectional:
                h_last_j = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h_last_j = h_n[-1]  # [B, hidden_dim]

            h_seq_list.append(h_seq_j.unsqueeze(1))   # [B, 1, T, out_dim]
            h_last_list.append(h_last_j.unsqueeze(1)) # [B, 1, out_dim]

        h_seq  = torch.cat(h_seq_list,  dim=1)  # [B, N, T, out_dim]
        h_last = torch.cat(h_last_list, dim=1)  # [B, N, out_dim]

        return h_seq, h_last