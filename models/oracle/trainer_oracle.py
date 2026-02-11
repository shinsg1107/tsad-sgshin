import os
import torch
import torch.nn.functional as F

from trainer import Trainer, prepare_inputs


class OracleADTrainer(Trainer):
    """
    OracleAD trainer that:
      1) builds SLS each epoch from c_star (mean D over all train windows in the epoch)
      2) uses deviation loss from START_EPOCH onward (when SLS is available)
      3) computes pred/recon losses in the trainer from model outputs:
           pred_loss  = MSE(x_hat_next, y_next)
           recon_loss = MSE(x_hat_past, x_past_true)
    """

    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.start_sls_epoch = int(self.cfg.ORACLEAD.SLS.START_EPOCH)
        self.lambda_recon = float(self.cfg.ORACLEAD.LAMBDA_RECON)
        self.lambda_dev = float(self.cfg.ORACLEAD.LAMBDA_DEV)

        # If your paper uses L2 (not squared) for D_ij, set this False.
        # Your current SLS builder uses squared L2 distance (||a-b||^2).
        self.use_squared_l2 = bool(getattr(self.cfg.ORACLEAD.SLS, "USE_SQUARED_L2", True))

        # SLS state (reference template)
        self.sls = None  # [N,N] on CPU; moved to GPU when used
        self._reset_sls_accum()

        sls_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "sls_latest.pt")
        if os.path.isfile(sls_path):
            self.sls = torch.load(sls_path, map_location="cpu")
            print(f"[OracleADTrainer] Loaded SLS from {sls_path} (shape={tuple(self.sls.shape)})")

    def _reset_sls_accum(self):
        self.sls_sum = None   # [N,N]
        self.sls_count = 0    # total windows M

    @staticmethod
    def _pairwise_sq_l2(c_star: torch.Tensor) -> torch.Tensor:
        """
        c_star: [B,N,D]
        return: [B,N,N] with squared L2 distances
        """
        x2 = (c_star * c_star).sum(dim=-1, keepdim=True)          # [B,N,1]
        prod = torch.matmul(c_star, c_star.transpose(1, 2))       # [B,N,N]
        dist2 = x2 + x2.transpose(1, 2) - 2.0 * prod
        return dist2.clamp_min_(0.0)

    def _pairwise_l2(self, c_star: torch.Tensor) -> torch.Tensor:
        """
        c_star: [B,N,D]
        return: [B,N,N] with L2 distances (not squared)
        """
        dist2 = self._pairwise_sq_l2(c_star)
        return (dist2 + 1e-12).sqrt()

    def _accumulate_sls(self, D_batch: torch.Tensor):
        """
        D_batch: [B,N,N]
        Accumulate sum over batch to build epoch mean SLS.
        """
        with torch.no_grad():
            D_sum = D_batch.detach().sum(dim=0)  # [N,N]
            self.sls_sum = D_sum if self.sls_sum is None else (self.sls_sum + D_sum)
            self.sls_count += int(D_batch.size(0))

    def _finalize_and_save_sls(self):
        """
        At end of epoch: SLS = (1/M) sum_k D(k)
        Save and set self.sls for next epoch.
        """
        if self.sls_sum is None or self.sls_count == 0:
            print("[OracleADTrainer] Warning: No D accumulated. SLS not updated.")
            return

        sls = (self.sls_sum / float(self.sls_count)).cpu()  # [N,N] on CPU
        self.sls = sls

        if bool(getattr(self.cfg.ORACLEAD.SLS, "SAVE_EVERY_EPOCH", True)):
            ckpt_dir = self.cfg.TRAIN.CHECKPOINT_DIR
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(sls, os.path.join(ckpt_dir, f"sls_epoch_{self.cur_epoch}.pt"))
            torch.save(sls, os.path.join(ckpt_dir, "sls_latest.pt"))

    def _unwrap_x(self, inputs):
        inputs = prepare_inputs(inputs)
        if isinstance(inputs, (tuple, list)):
            return inputs[0]  # (x, label) 형태면 x만
        return inputs

    def train_epoch(self):
        # epoch 시작: 이번 epoch에서 dev loss를 쓸지 결정
        self.use_dev_loss = (self.cur_epoch >= self.start_sls_epoch) and (self.sls is not None)

        self._reset_sls_accum()
        super().train_epoch()
        self._finalize_and_save_sls()

    def _compute_pred_recon_loss(self, out: dict):
        """
        out keys expected:
          x_hat_next: [B,N] (or [B,N,1] -> we squeeze)
          y_next:     [B,N]
          x_hat_past: [B,N,T] (or [B,N,T,1] -> we squeeze)
          x_past_true:[B,N,T]
        """
        x_hat_next = out["x_hat_next"]
        y_next = out["y_next"]
        if x_hat_next.dim() == 3 and x_hat_next.size(-1) == 1:
            x_hat_next = x_hat_next.squeeze(-1)  # [B,N]

        x_hat_past = out["x_hat_past"]
        x_past_true = out["x_past_true"]
        if x_hat_past.dim() == 4 and x_hat_past.size(-1) == 1:
            x_hat_past = x_hat_past.squeeze(-1)  # [B,N,T]

        # MSE (paper text uses L2 norms; MSE is fine up to scaling. If you want pure L2, change reduction.)
        pred_loss = F.mse_loss(x_hat_next, y_next)
        recon_loss = F.mse_loss(x_hat_past, x_past_true)
        return pred_loss, recon_loss, x_hat_next, x_hat_past

    def train_step(self, inputs):
        x = self._unwrap_x(inputs)
        out = self.model(x)

        # 1) compute losses from model outputs
        pred_loss, recon_loss, _, _ = self._compute_pred_recon_loss(out)
        c_star = out["c_star"]  # [B,N,D]

        # 2) build dissimilarity matrix D(k) and accumulate SLS (always)
        if self.use_squared_l2:
            D_batch = self._pairwise_sq_l2(c_star)  # [B,N,N]
        else:
            D_batch = self._pairwise_l2(c_star)     # [B,N,N]
        self._accumulate_sls(D_batch)

        # 3) deviation loss
        if self.use_dev_loss:
            sls = self.sls.to(D_batch.device, dtype=D_batch.dtype)  # [N,N]
            diff = D_batch - sls[None, :, :]                        # [B,N,N]
            N = diff.size(1)
            dev_loss = (diff.pow(2).sum(dim=(1, 2)) / float(N * N)).mean()
            total = pred_loss + self.lambda_recon * recon_loss + self.lambda_dev * dev_loss
        else:
            dev_loss = pred_loss.new_zeros(())
            total = pred_loss + self.lambda_recon * recon_loss

        # 4) optimize
        self.optimizer.zero_grad()
        total.backward()
        if self.cfg.SOLVER.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.GRADIENT_CLIP_NORM)
        self.optimizer.step()

        return {
            "metrics": (total, pred_loss, recon_loss, dev_loss),
            "losses":  (total, pred_loss, recon_loss, dev_loss),
        }

    @torch.no_grad()
    def eval_step(self, inputs):
        x = self._unwrap_x(inputs)
        out = self.model(x)

        pred_loss, recon_loss, _, _ = self._compute_pred_recon_loss(out)
        c_star = out["c_star"]

        if self.use_squared_l2:
            D_batch = self._pairwise_sq_l2(c_star)
        else:
            D_batch = self._pairwise_l2(c_star)

        use_dev = (self.cur_epoch >= self.start_sls_epoch) and (self.sls is not None)
        if use_dev:
            sls = self.sls.to(D_batch.device, dtype=D_batch.dtype)
            diff = D_batch - sls[None, :, :]
            N = diff.size(1)
            dev_loss = (diff.pow(2).sum(dim=(1, 2)) / float(N * N)).mean()
            total = pred_loss + self.lambda_recon * recon_loss + self.lambda_dev * dev_loss
        else:
            dev_loss = pred_loss.new_zeros(())
            total = pred_loss + self.lambda_recon * recon_loss

        return {
            "metrics": (total, pred_loss, recon_loss, dev_loss),
            "losses":  (total, pred_loss, recon_loss, dev_loss),
        }
