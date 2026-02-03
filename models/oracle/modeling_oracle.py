import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.LSTM_layer import LSTMEncoder, LSTMDecoder
from layers.pooling import LearnableAttnPooling
from layers.SelfAttention_Family import OracleAD_MHSA

class ORACLEAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg.DATA
        self.cfg_oraclead = cfg.ORACLEAD

        self.encoder = self._init_encoder()
        self.pool = self._init_pool()
        self.mhsa = self._init_mhsa()
        self.decoder = self._init_decoder()

        self.w_recon = float(getattr(self.cfg_oraclead, "RECON_WEIGHT", 1.0))
        self.w_pred = float(getattr(self.cfg_oraclead, "PRED_WEIGHT", 1.0))

        # metric/loss names (Trainer가 읽을 수 있게)
        # cfg.ORACLEAD.METRIC_NAMES / LOSS_NAMES 를 쓰는 구조와 호환

    def _init_encoder(self):
        return LSTMEncoder(self.cfg)

    def _init_pool(self):
        dim = int(self.cfg_oraclead.LSTM_ENCODER.HIDDEN_DIM) * (2 if bool(getattr(self.cfg_oraclead.LSTM_ENCODER, "BIDIRECTIONAL", False)) else 1)
        return LearnableAttnPooling(dim=dim)

    def _init_mhsa(self):
        return OracleAD_MHSA(self.cfg)

    def _init_decoder(self):
        return LSTMDecoder(self.cfg)

    def _standardize_input(self, x: torch.Tensor):
        """
        Accept common window formats and convert to:
          x_past: [B, N, T, 1],  y_next: [B, N]  where T = L-1
        Expected original window length L = cfg.DATA.WIN_SIZE
        """
        # common cases:
        #   (a) x: [B, L, N]
        #   (b) x: [B, N, L]
        assert x.dim() == 3, f"Expected 3D window input, got {x.shape}"

        B = x.size(0)
        N = int(self.cfg_data.N_VAR)
        L = int(self.cfg_data.WIN_SIZE)

        if x.size(1) == L and x.size(2) == N:
            # [B, L, N]
            x_ln = x
        elif x.size(1) == N and x.size(2) == L:
            # [B, N, L] -> [B, L, N]
            x_ln = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unexpected input shape {x.shape}. Expect [B,L,N] or [B,N,L] with L={L}, N={N}")

        x_past_ln = x_ln[:, :-1, :]  # [B, L-1, N]
        y_next = x_ln[:, -1, :]      # [B, N]

        # -> [B, N, T, 1]
        x_past = x_past_ln.transpose(1, 2).unsqueeze(-1).contiguous()  # [B,N,T,1]
        return x_past, y_next

    def forward(self, x):
        x_past, y_next = self._standardize_input(x)  # x_past: [B,N,T,1], y_next: [B,N]
        B, N, T, _ = x_past.shape

        # 1) per-variable temporal encoding
        h_seq, _ = self.encoder(x_past)         # [B,N,T,D]

        # 2) learnable attention pooling over time
        c, alpha_time = self.pool(h_seq)        # c: [B,N,D], alpha_time: [B,N,T]

        # 3) MHSA over variables
        c_star, attn_var = self.mhsa(c)         # [B,N,D], [B,H,N,N]

        # 4) per-variable decoder (recon + next pred)
        x_hat_past, x_hat_next = self.decoder(c_star)  # [B,N,T,1], [B,N,1]

        # align shapes for losses
        x_hat_past = x_hat_past.squeeze(-1)     # [B,N,T]
        x_hat_next = x_hat_next.squeeze(-1)     # [B,N]

        outputs = {
            "x_hat_past": x_hat_past.detach(),
            "x_hat_next": x_hat_next.detach(),
        }
        return outputs
