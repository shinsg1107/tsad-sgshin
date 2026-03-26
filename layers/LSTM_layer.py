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


# class LSTMDecoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.cfg_dec = cfg.ORACLEAD.DECODER

#         self.n_vars    = int(cfg.DATA.N_VAR)
#         self.dim       = int(self.cfg_dec.DIM)
#         self.hidden_dim= int(self.cfg_dec.HIDDEN_DIM)
#         self.num_layers= int(getattr(self.cfg_dec, "NUM_LAYERS", 1))
#         self.dropout   = float(getattr(self.cfg_dec, "DROPOUT", 0.0))
#         self.bias      = bool(getattr(self.cfg_dec, "BIAS", True))
#         self.out_dim   = int(getattr(self.cfg_dec, "OUT_DIM", 1))
#         self.T         = int(getattr(self.cfg_dec, "PAST_LEN", cfg.DATA.WIN_SIZE - 1))
#         self.L         = self.T + 1  # full window length

#         def make_lstm():
#             return nn.LSTM(
#                 input_size=self.out_dim,
#                 hidden_size=self.hidden_dim,
#                 num_layers=self.num_layers,
#                 batch_first=True,
#                 dropout=self.dropout if self.num_layers > 1 else 0.0,
#                 bias=self.bias,
#             )

#         self.lstm_list   = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])

#         # 학습 가능한 hidden/cell 초기화 projection
#         self.init_h_list = nn.ModuleList([
#             nn.Linear(self.dim, self.num_layers * self.hidden_dim)
#             for _ in range(self.n_vars)
#         ])
#         self.init_c_list = nn.ModuleList([
#             nn.Linear(self.dim, self.num_layers * self.hidden_dim)
#             for _ in range(self.n_vars)
#         ])

#         self.out_list = nn.ModuleList([
#             nn.Linear(self.hidden_dim, self.out_dim)
#             for _ in range(self.n_vars)
#         ])

#     def forward(self, c_star: torch.Tensor):
#         B, N, D = c_star.shape
#         assert N == self.n_vars
#         assert D == self.dim

#         x_hat_past_list = []
#         x_hat_next_list = []

#         for i in range(N):
#             ci = c_star[:, i, :]  # [B, D]
#             z = torch.zeros(B, self.L, self.out_dim, device=ci.device, dtype=ci.dtype)

#             # [B, num_layers * hidden_dim] → [B, num_layers, hidden_dim] → [num_layers, B, hidden_dim]
#             h0 = torch.tanh(self.init_h_list[i](ci))
#             h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

#             c0 = torch.tanh(self.init_c_list[i](ci))
#             c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

#             Y, _ = self.lstm_list[i](z, (h0, c0))  # [B, L, hidden_dim]

#             O = self.out_list[i](Y).squeeze(-1)     # [B, L]

#             recon_i = O[:, :self.T]                 # [B, T]
#             next_i  = O[:, self.T]                  # [B]

#             x_hat_past_list.append(recon_i.unsqueeze(1))             # [B, 1, T]
#             x_hat_next_list.append(next_i.unsqueeze(1).unsqueeze(-1))  # [B, 1, 1]

#         x_hat_past = torch.cat(x_hat_past_list, dim=1).unsqueeze(-1)  # [B, N, T, 1]
#         x_hat_next = torch.cat(x_hat_next_list, dim=1)                # [B, N, 1]
        
#         return x_hat_past, x_hat_next


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

        # 1-step lag
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
        self.L         = self.T + 1

        # ------------------------------------------------------------------ #
        # Causal graph 설정 (encoder와 동일한 config 공유)
        # ------------------------------------------------------------------ #
        self.causal_mode    : Optional[str]             = None
        self.causal_alpha   : float                     = 0.0
        self.topk           : int                       = 3
        self.parent_indices : Optional[List[List[int]]] = None
        self.proj_list      : Optional[nn.ModuleList]   = None
        self.register_buffer("causal_weights", None)
        self.causal_residual = bool(getattr(
            self.cfg_dec, "CAUSAL_RESIDUAL", False
        ))

        # causal_residual이 True이고 CAUSAL_ENCODER도 활성화된 경우만 causal 설정
        cfg_causal = getattr(cfg.ORACLEAD, "CAUSAL_ENCODER", None)
        causal_enable = (
            self.causal_residual and
            cfg_causal is not None and
            bool(getattr(cfg_causal, "ENABLE", False))
        )

        enc_hidden_dim = int(cfg.ORACLEAD.LSTM_ENCODER.HIDDEN_DIM)

        if causal_enable:
            path = getattr(cfg_causal, "GRAPH_PATH", "")
            if path and os.path.isfile(path):
                cg = load_causal_graph(
                    path,
                    normalize=bool(getattr(cfg_causal, "NORMALIZE", True))
                )
                assert cg.shape == (self.n_vars, self.n_vars)

                self.causal_mode  = getattr(cfg_causal, "MODE", "linear")
                self.causal_alpha = float(getattr(cfg_causal, "ALPHA", 0.3))
                threshold         = float(getattr(cfg_causal, "THRESHOLD", 0.1))
                self.topk         = int(getattr(cfg_causal, "TOPK", 3))
                d_proj            = int(getattr(cfg_causal, "PROJ_DIM", 16))

                self.register_buffer("causal_weights", torch.from_numpy(cg))

                # 변수별 parent 파악 (encoder와 동일 로직)
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

                # decoder input fusion:
                # [zero(out_dim), h_enc_j^t, parent_h_enc^{t-1}] → Linear → lstm_input_size
                # encoder hidden dim으로 입력 구성
                max_p = max(len(p) for p in self.parent_indices)

                if self.causal_mode == 'max_pad':
                    # zero + h_enc_j + parent h_encs (padded)
                    fusion_in = self.out_dim + enc_hidden_dim * (1 + max_p)
                elif self.causal_mode == 'topk':
                    fusion_in = self.out_dim + enc_hidden_dim * (1 + self.topk)
                elif self.causal_mode == 'weighted':
                    # zero + h_enc_j + weighted_sum_parent (scalar→enc_hidden_dim)
                    fusion_in = self.out_dim + enc_hidden_dim * 2
                elif self.causal_mode == 'linear':
                    fusion_in = self.out_dim + enc_hidden_dim * (1 + len(
                        max(self.parent_indices, key=len)
                    ))

                # 변수별 fusion Linear
                # max_pad/weighted: 공통 크기 → 하나의 Linear
                # topk/linear: 변수별 크기 다를 수 있음 → ModuleList
                if self.causal_mode in ('max_pad', 'weighted'):
                    self.fusion_proj = nn.Linear(fusion_in, self.out_dim)
                elif self.causal_mode == 'topk':
                    self.fusion_proj = nn.Linear(fusion_in, self.out_dim)
                elif self.causal_mode == 'linear':
                    self.fusion_proj = nn.ModuleList([
                        nn.Linear(
                            self.out_dim + enc_hidden_dim * (1 + len(self.parent_indices[j])),
                            self.out_dim
                        )
                        for j in range(self.n_vars)
                    ])

                print(f"[LSTMDecoder] causal_mode={self.causal_mode}, "
                      f"enc_hidden_dim={enc_hidden_dim}")
            else:
                print(f"[LSTMDecoder] CAUSAL path not found: {path}")
        else:
            # causal_residual=False 또는 CAUSAL_ENCODER.ENABLE=False
            self.causal_mode = None
            print(f"[LSTMDecoder] causal_residual={self.causal_residual}, "
                  f"causal_enable={causal_enable} → 기존 decoder 사용")

        # ------------------------------------------------------------------ #
        # LSTM / projection
        # ------------------------------------------------------------------ #
        def make_lstm():
            return nn.LSTM(
                input_size=self.out_dim,   # fusion 후 항상 out_dim
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bias=self.bias,
            )

        self.lstm_list = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])

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

    # ------------------------------------------------------------------ #
    # timestep별 decoder input 구성
    # ------------------------------------------------------------------ #
    def _build_dec_input(
        self,
        z_t: torch.Tensor,         # [B, out_dim]  zero
        h_enc: torch.Tensor,       # [B, N, T, enc_hidden]
        h_enc_lag: torch.Tensor,   # [B, N, T, enc_hidden]  1-step lagged
        j: int,
        t: int
    ) -> torch.Tensor:
        """
        [z^t, h_enc_j^t, parent_h_enc^{t-1}] → Linear → [B, out_dim]
        """
        h_j = h_enc[:, j, t, :]      # [B, enc_hidden]

        if self.causal_mode == 'max_pad':
            max_p   = max(len(p) for p in self.parent_indices)
            parents = self.parent_indices[j]
            parts   = [z_t, h_j]
            for p in parents:
                parts.append(h_enc_lag[:, p, t, :])  # [B, enc_hidden]
            n_pad = max_p - len(parents)
            if n_pad > 0:
                parts.append(torch.zeros(
                    z_t.size(0), self.causal_weights.size(0) * n_pad // self.n_vars,
                    device=z_t.device, dtype=z_t.dtype
                ))
            x_cat = torch.cat(parts, dim=-1)
            return self.fusion_proj(x_cat)  # [B, out_dim]

        elif self.causal_mode == 'topk':
            parents = self.parent_indices[j]
            parts   = [z_t, h_j]
            for p in parents:
                parts.append(h_enc_lag[:, p, t, :])
            n_pad = self.topk - len(parents)
            if n_pad > 0:
                enc_h = h_enc.size(-1)
                parts.append(torch.zeros(
                    z_t.size(0), enc_h * n_pad,
                    device=z_t.device, dtype=z_t.dtype
                ))
            x_cat = torch.cat(parts, dim=-1)
            return self.fusion_proj(x_cat)  # [B, out_dim]

        elif self.causal_mode == 'weighted':
            # weighted sum of parent enc hiddens
            weights   = self.causal_weights[:, j]             # [N]
            h_lag_sq  = h_enc_lag[:, :, t, :]                 # [B, N, enc_hidden]
            parent_h  = torch.einsum('n,bnd->bd', weights, h_lag_sq)  # [B, enc_hidden]
            x_cat     = torch.cat([z_t, h_j, parent_h], dim=-1)
            return self.fusion_proj(x_cat)

        elif self.causal_mode == 'linear':
            parents = self.parent_indices[j]
            parts   = [z_t, h_j]
            for p in parents:
                parts.append(h_enc_lag[:, p, t, :])
            x_cat = torch.cat(parts, dim=-1)
            return self.fusion_proj[j](x_cat)  # 변수별 Linear

        return z_t  # causal 없으면 그대로

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        c_star: torch.Tensor,
        h_enc: Optional[torch.Tensor] = None,   # [B, N, T, enc_hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        c_star: [B, N, D]
        h_enc:  [B, N, T, enc_hidden]  encoder hidden sequence (causal 사용 시)
        """
        B, N, D = c_star.shape
        assert N == self.n_vars
        assert D == self.dim

        # 1-step lagged encoder hidden
        h_enc_lag = None
        if self.causal_mode is not None and h_enc is not None:
            h_enc_lag = torch.zeros_like(h_enc)
            h_enc_lag[:, :, 1:, :] = h_enc[:, :, :-1, :]  # t=0은 0

        x_hat_past_list = []
        x_hat_next_list = []

        for i in range(N):
            ci = c_star[:, i, :]  # [B, D]

            # hidden/cell 초기화
            h0 = torch.tanh(self.init_h_list[i](ci))
            h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            c0 = torch.tanh(self.init_c_list[i](ci))
            c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

            if self.causal_mode is not None and h_enc is not None:
                # timestep별로 causal input 구성 후 LSTM 순차 실행
                h_state = (h0, c0)
                Y_list  = []

                for t in range(self.L):
                    z_t = torch.zeros(B, self.out_dim, device=ci.device, dtype=ci.dtype)

                    # t < T면 h_enc 참조, t == T (next pred)면 마지막 h_enc 사용
                    t_enc = min(t, self.T - 1)
                    dec_in_t = self._build_dec_input(
                        z_t, h_enc, h_enc_lag, i, t_enc
                    )  # [B, out_dim]

                    # LSTM 한 step
                    dec_in_t = dec_in_t.unsqueeze(1)          # [B, 1, out_dim]
                    y_t, h_state = self.lstm_list[i](dec_in_t, h_state)
                    Y_list.append(y_t)                        # [B, 1, hidden_dim]

                Y = torch.cat(Y_list, dim=1)                  # [B, L, hidden_dim]

            else:
                # 기존 방식: zero input 전체를 한번에
                z = torch.zeros(B, self.L, self.out_dim, device=ci.device, dtype=ci.dtype)
                Y, _ = self.lstm_list[i](z, (h0, c0))        # [B, L, hidden_dim]

            O = self.out_list[i](Y).squeeze(-1)               # [B, L]

            recon_i = O[:, :self.T]                           # [B, T]
            next_i  = O[:, self.T]                            # [B]

            x_hat_past_list.append(recon_i.unsqueeze(1))
            x_hat_next_list.append(next_i.unsqueeze(1).unsqueeze(-1))

        x_hat_past = torch.cat(x_hat_past_list, dim=1).unsqueeze(-1)  # [B, N, T, 1]
        x_hat_next = torch.cat(x_hat_next_list, dim=1)                # [B, N, 1]

        return x_hat_past, x_hat_next