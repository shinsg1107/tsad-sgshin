import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    def __init__(self,cfg):
        super(LSTMEncoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.ORACLEAD.LSTM_ENCODER

        self.n_vars = cfg.DATA.N_VAR
        self.input_dim = self.cfg_model.INPUT_DIM
        self.hidden_dim = self.cfg_model.HIDDEN_DIM
        self.num_layers = self.cfg_model.NUM_LAYERS
        self.dropout = self.cfg_model.DROPOUT
        self.bidirectional = self.cfg_model.BIDIRECTIONAL
        self.shared = self.cfg_model.SHARED #LSTM 공유 여부

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        def make_lstm():
            return nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
                batch_first=True,  # [B, T, C]
            )

        if self.shared:
            self.lstm = make_lstm()
        else:
            self.lstm_list = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, T]  (or [B, N, T, C] if input_dim>1)
        """
        if x.dim() == 3:
            # [B, N, T] -> [B, N, T, 1]
            x = x.unsqueeze(-1)

        B, N, T, C = x.shape
        assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got {C}"

        if self.shared:
            # Flatten variables into batch: [B*N, T, C]
            x_bn = x.reshape(B * N, T, C)
            h_seq_bn, (h_n, c_n) = self.lstm(x_bn)  # h_seq_bn: [B*N, T, out_dim]
            h_seq = h_seq_bn.reshape(B, N, T, self.out_dim)
            # last layer hidden: [num_layers * num_dir, B*N, hidden_dim]
            # take last layer (and concat dirs already in out_dim for h_seq)
            if self.bidirectional:
                # last layer forward/backward hidden concat
                # h_n shape: [num_layers*2, B*N, hidden_dim]
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B*N, 2*hidden_dim]
            else:
                h_last = h_n[-1]  # [B*N, hidden_dim]
            h_last = h_last.reshape(B, N, self.out_dim)
        else:
            # Per-variable LSTM
            h_seq_list = []
            h_last_list = []
            for i in range(N):
                xi = x[:, i, :, :]              # [B, T, C]
                h_seq_i, (h_n, c_n) = self.lstm_list[i](xi)  # [B, T, out_dim]
                if self.bidirectional:
                    h_last_i = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2*hidden_dim]
                else:
                    h_last_i = h_n[-1]  # [B, hidden_dim]
                h_seq_list.append(h_seq_i.unsqueeze(1))   # [B, 1, T, out_dim]
                h_last_list.append(h_last_i.unsqueeze(1)) # [B, 1, out_dim]
            h_seq = torch.cat(h_seq_list, dim=1)   # [B, N, T, out_dim]
            h_last = torch.cat(h_last_list, dim=1) # [B, N, out_dim]

        return h_seq, h_last
    
class LSTMDecoder(nn.Module):
    """
    OracleAD-style decoder(s):
      Dec_i(c_i*) -> (xhat_i^{1:L-1}, xhat_i^L)

    Input:
      c_star: [B, N, D]
    Output:
      x_hat_past: [B, N, T, C_out]   (reconstruct past window)
      x_hat_next: [B, N, C_out]      (predict next value)

    Notes:
      - T should match (L-1) = cfg.ORACLE.WIN_SIZE-1 or cfg.DATA.WIN_SIZE-1 depending on your setup.
      - Uses an LSTM that is driven by repeated context vector c* across time.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_dec = cfg.ORACLEAD.DECODER

        self.n_vars = int(cfg.DATA.N_VAR)
        self.dim = int(self.cfg_dec.DIM)                 # D (must match c* dim)
        self.hidden_dim = int(self.cfg_dec.HIDDEN_DIM)   # decoder hidden
        self.num_layers = int(getattr(self.cfg_dec, "NUM_LAYERS", 1))
        self.dropout = float(getattr(self.cfg_dec, "DROPOUT", 0.0))
        self.bias = bool(getattr(self.cfg_dec, "BIAS", True))
        self.bidirectional = bool(getattr(self.cfg_dec, "BIDIRECTIONAL", False))
        self.shared = bool(getattr(self.cfg_dec, "SHARED", False))

        # output dim per variable (usually 1 for a single sensor value)
        self.out_dim = int(getattr(self.cfg_dec, "OUT_DIM", 1))

        # past length T = (L-1)
        # OracleAD paper uses window length L, and reconstructs 1..L-1 and predicts L.
        self.T = int(getattr(self.cfg_dec, "PAST_LEN", cfg.DATA.WIN_SIZE - 1))

        # LSTM input size is D because we feed c* (context) each step
        lstm_in = self.dim
        lstm_out = self.hidden_dim * (2 if self.bidirectional else 1)

        def make_lstm():
            return nn.LSTM(
                input_size=lstm_in,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bias=self.bias,
                bidirectional=self.bidirectional,
            )

        if self.shared:
            self.lstm = make_lstm()
            self.head_recon = nn.Linear(lstm_out, self.out_dim)
            self.head_next = nn.Linear(lstm_out, self.out_dim)
        else:
            self.lstm_list = nn.ModuleList([make_lstm() for _ in range(self.n_vars)])
            self.head_recon_list = nn.ModuleList([nn.Linear(lstm_out, self.out_dim) for _ in range(self.n_vars)])
            self.head_next_list = nn.ModuleList([nn.Linear(lstm_out, self.out_dim) for _ in range(self.n_vars)])

    def forward(self, c_star: torch.Tensor):
        """
        c_star: [B, N, D]
        """
        B, N, D = c_star.shape
        assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"
        assert D == self.dim, f"Expected D={self.dim}, got {D}"

        # We drive the decoder LSTM with repeated context vector across time:
        # dec_in_i: [B, T, D] where each time step input is c_i*
        # This is a common way to decode a sequence from a fixed context.
        if self.shared:
            dec_in = c_star.unsqueeze(2).expand(B, N, self.T, D)      # [B, N, T, D]
            dec_in = dec_in.reshape(B * N, self.T, D)                # [B*N, T, D]

            h_seq, (h_n, _) = self.lstm(dec_in)                      # h_seq: [B*N, T, lstm_out]
            recon = self.head_recon(h_seq)                           # [B*N, T, out_dim]
            next_ = self.head_next(h_seq[:, -1, :])                  # [B*N, out_dim]

            x_hat_past = recon.reshape(B, N, self.T, self.out_dim)   # [B, N, T, out_dim]
            x_hat_next = next_.reshape(B, N, self.out_dim)           # [B, N, out_dim]
            return x_hat_past, x_hat_next

        # Per-variable decoders (paper-faithful)
        x_hat_past_list = []
        x_hat_next_list = []

        for i in range(N):
            ci = c_star[:, i, :]                                     # [B, D]
            dec_in_i = ci.unsqueeze(1).expand(B, self.T, D)          # [B, T, D]

            h_seq_i, (h_n, _) = self.lstm_list[i](dec_in_i)          # [B, T, lstm_out]
            recon_i = self.head_recon_list[i](h_seq_i)               # [B, T, out_dim]
            next_i = self.head_next_list[i](h_seq_i[:, -1, :])       # [B, out_dim]

            x_hat_past_list.append(recon_i.unsqueeze(1))             # [B, 1, T, out_dim]
            x_hat_next_list.append(next_i.unsqueeze(1))              # [B, 1, out_dim]

        x_hat_past = torch.cat(x_hat_past_list, dim=1)               # [B, N, T, out_dim]
        x_hat_next = torch.cat(x_hat_next_list, dim=1)               # [B, N, out_dim]
        return x_hat_past, x_hat_next
