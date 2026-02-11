import torch
import torch.nn as nn

from layers.LSTM_layer import LSTMEncoder, LSTMDecoder
from layers.pooling import LearnableAttnPooling
from layers.SelfAttention_Family import OracleAD_MHSA

class ORACLEAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg.DATA
        self.cfg_oraclead = cfg.ORACLEAD

        self.encoder = LSTMEncoder(cfg)
        dim = int(self.cfg_oraclead.LSTM_ENCODER.HIDDEN_DIM) * (
            2 if bool(getattr(self.cfg_oraclead.LSTM_ENCODER, "BIDIRECTIONAL", False)) else 1
        )
        self.pool = LearnableAttnPooling(dim=dim)
        self.mhsa = OracleAD_MHSA(cfg)
        self.decoder = LSTMDecoder(cfg)

    def _standardize_input(self, x: torch.Tensor):
        # x: [B,L,N] or [B,N,L]
        assert x.dim() == 3, f"Expected 3D window input, got {x.shape}"
        N = int(self.cfg_data.N_VAR)
        L = int(self.cfg_data.WIN_SIZE)

        if x.size(1) == L and x.size(2) == N:
            x_ln = x
        elif x.size(1) == N and x.size(2) == L:
            x_ln = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unexpected input shape {x.shape}. Expect [B,L,N] or [B,N,L] with L={L}, N={N}")

        x_past_ln = x_ln[:, :-1, :]  # [B, L-1, N]
        y_next = x_ln[:, -1, :]      # [B, N]

        x_past = x_past_ln.transpose(1, 2).unsqueeze(-1).contiguous()  # [B,N,T,1], T=L-1
        return x_past, y_next

    def forward(self, x):
        x_past, y_next = self._standardize_input(x)  # x_past: [B,N,T,1], y_next: [B,N]
        B, N, T, _ = x_past.shape

        # 1) per-variable LSTM encoder
        h_seq, _ = self.encoder(x_past)              # [B,N,T,D]

        # 2) attention pooling over time
        c, alpha_time = self.pool(h_seq)             # c: [B,N,D], alpha_time: [B,N,T]

        # 3) MHSA over variables
        c_star, attn_var = self.mhsa(c)              # c_star: [B,N,D]

        # 4) decoder: reconstruct past + predict next
        x_hat_past, x_hat_next = self.decoder(c_star)  # [B,N,T,1], [B,N,1]
        x_hat_past = x_hat_past.squeeze(-1)            # [B,N,T]
        x_hat_next = x_hat_next.squeeze(-1)            # [B,N]
        
        outputs = {
            "x_hat_past": x_hat_past,     # recon target: x_past.squeeze(-1)
            "x_hat_next": x_hat_next,     # pred target: y_next
            "c_star": c_star,             # SLS / deviation 계산용
            "y_next": y_next,             # trainer 편의
            "x_past_true": x_past.squeeze(-1),  # trainer 편의
            "alpha_time": alpha_time,     # optional
            "attn_var": attn_var,         # optional
        }
        return outputs
