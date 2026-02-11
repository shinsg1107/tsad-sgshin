import os
import torch

class ScorerOracleAD:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model.eval()
        sls_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "sls_latest.pt")
        self.sls = torch.load(sls_path, map_location="cpu") if os.path.isfile(sls_path) else None

    @staticmethod
    def pairwise_sq_l2(c_star: torch.Tensor) -> torch.Tensor:
        """
        c_star: [B, N, D]
        return: [B, N, N] where D_ij = ||c_i - c_j||_2^2
        """
        x2 = (c_star ** 2).sum(dim=-1, keepdim=True)                 # [B,N,1]
        prod = c_star @ c_star.transpose(1, 2)                       # [B,N,N]
        dist2 = x2 + x2.transpose(1, 2) - 2.0 * prod                 # [B,N,N]
        return dist2.clamp_min(0.0)

    @staticmethod
    def frob_norm(mat: torch.Tensor) -> torch.Tensor:
        """
        mat: [B, N, N] -> [B]
        """
        return torch.sqrt((mat ** 2).sum(dim=(1, 2)) + 1e-12)

    @torch.no_grad()
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: window input (예: [B,L,N] 또는 [B,N,L])
        return: A_score^t for each window end-time, shape [B]
        """
        out = self.model(x)
        x_hat_next = out["x_hat_next"]      # [B,N]
        c_star     = out["c_star"]          # [B,N,D]

        # true x^t 만들기 (모델 out에 없으면 여기서 복원)
        if "x_true_next" in out:
            x_true_next = out["x_true_next"]  # [B,N]
        else:
            L = self.cfg.DATA.WIN_SIZE
            N = self.cfg.DATA.N_VAR
            if x.dim() == 3 and x.size(1) == L and x.size(2) == N:
                x_ln = x
            else:
                x_ln = x.transpose(1, 2).contiguous()  # [B,L,N] 가정
            x_true_next = x_ln[:, -1, :]               # [B,N]

        # (15) prediction score: mean_i |x_i^t - xhat_i^t|
        P = (x_true_next - x_hat_next).abs().mean(dim=-1)  # [B]

        # (16) deviation score: ||D^t - SLS||_F
        if self.sls is None:
            D_score = torch.zeros_like(P)
        else:
            D_t = self.pairwise_sq_l2(c_star)  # [B,N,N]
            sls = self.sls.to(D_t.device, dtype=D_t.dtype)  # [N,N]
            D_score = self.frob_norm(D_t - sls[None])        # [B]

        A = P * D_score 
        return A
