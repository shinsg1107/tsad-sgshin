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
        assert x.dim() == 3
        N = int(self.cfg_data.N_VAR)
        L = int(self.cfg_data.WIN_SIZE)

        causal_extra = bool(getattr(
            getattr(self.cfg_oraclead, "CAUSAL_ENCODER", None),
            "USE_EXTRA_STEP", False
        ))

        if causal_extra:
            # x: [B, L+1, N]
            # x[:,0,:] = t=-1 (lag base)
            # x[:,1:,:] = t=0~L-1
            x_lag_base = x[:, 0, :]   # [B, N] window 이전 값
            x = x[:, 1:, :]           # [B, L, N]
        else:
            x_lag_base = None

        # shape 확인
        if x.size(1) == L and x.size(2) == N:
            x_ln = x
        elif x.size(1) == N and x.size(2) == L:
            x_ln = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unexpected shape {x.shape}")

        x_past_ln = x_ln[:, :-1, :]   # [B, L-1, N]
        y_next    = x_ln[:, -1, :]    # [B, N]
        x_past    = x_past_ln.transpose(1,2).unsqueeze(-1)  # [B, N, T, 1]

        # x_lag 구성
        if causal_extra and x_lag_base is not None:
            x_lag = torch.zeros_like(x_past)
            x_lag[:, :, 1:, :] = x_past[:, :, :-1, :]              # t>=1: 이전 step
            x_lag[:, :, 0, :]  = x_lag_base.unsqueeze(-1)           # t=0: window 이전 값 ✅
        else:
            x_lag = None  # encoder에서 자동 생성 (t=0은 0 패딩)

        return x_past, y_next, x_lag

    def forward(self, x):
        x_past, y_next, x_lag = self._standardize_input(x)
        
        # 1) per-variable LSTM encoder
        h_seq, _ = self.encoder(x_past, x_lag=x_lag)             # [B,N,T,D]

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
