import os
import torch
import torch.nn.functional as F

from trainer import Trainer, prepare_inputs
from utils.misc import mkdir


class OracleADTrainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.start_sls_epoch = int(self.cfg.ORACLEAD.SLS.START_EPOCH)
        self.lambda_recon = float(self.cfg.ORACLEAD.LAMBDA_RECON)
        self.lambda_dev = float(self.cfg.ORACLEAD.LAMBDA_DEV)

        # SLS state (reference template)
        self.sls = None  # [N,N] on device when used

        # accumulator for building SLS at end of each epoch
        self._reset_sls_accum()

        # (선택) 이전 학습 이어가기용: sls_latest.pt 있으면 로드
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
        return: D_batch [B,N,N] with squared L2 distances
        Efficient: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        """
        # [B,N,1]
        x2 = (c_star * c_star).sum(dim=-1, keepdim=True)
        # [B,N,N]
        prod = torch.matmul(c_star, c_star.transpose(1, 2))
        dist2 = x2 + x2.transpose(1, 2) - 2.0 * prod
        return dist2.clamp_min_(0.0)

    def _accumulate_sls(self, D_batch: torch.Tensor):
        """
        D_batch: [B,N,N]
        Accumulate sum over batch to build epoch mean SLS.
        """
        with torch.no_grad():
            D_sum = D_batch.detach().sum(dim=0)  # [N,N]
            if self.sls_sum is None:
                self.sls_sum = D_sum
            else:
                self.sls_sum += D_sum
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
        self.sls = sls  # keep CPU copy; moved to GPU when used

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
        use_dev = (self.cur_epoch >= self.start_sls_epoch) and (self.sls is not None)
        self.use_dev_loss = use_dev

        # 이번 epoch 끝에서 업데이트할 SLS 누적기 초기화
        self._reset_sls_accum()

        # 원래 Trainer loop 수행 (각 batch마다 train_step 호출)
        super().train_epoch()

        # epoch 끝: 이번 epoch 데이터로 SLS 구축/저장
        self._finalize_and_save_sls()

    def train_step(self, inputs):
        x = self._unwrap_x(inputs)
        out = self.model(x)

        pred_loss = out["pred_loss"]     # scalar tensor (detach X)
        recon_loss = out["recon_loss"]   # scalar tensor (detach X)
        c_star = out["c_star"]           # [B,N,D] (detach X)

        # D(k) 계산 + SLS 누적 (항상 누적: 1 epoch 끝에 SLS 만들기 위해)
        D_batch = self._pairwise_sq_l2(c_star)    # [B,N,N]
        self._accumulate_sls(D_batch)

        # deviation/dev loss (첫 epoch에는 0)
        if self.use_dev_loss:
            sls = self.sls.to(D_batch.device, dtype=D_batch.dtype)  # [N,N]
            diff = D_batch - sls[None, :, :]
            N = diff.size(1)
            dev_per_sample = (diff * diff).sum(dim=(1, 2)) / float(N * N)  # [B]
            dev_loss = dev_per_sample.mean()
            total = pred_loss + self.lambda_recon * recon_loss + self.lambda_dev * dev_loss
        else:
            dev_loss = pred_loss.new_zeros(())  # meter 길이 유지
            total = pred_loss + self.lambda_recon * recon_loss

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

        pred_loss = out["pred_loss"]
        recon_loss = out["recon_loss"]
        c_star = out["c_star"]

        # eval에서는 SLS를 업데이트하지 않음 (논문도 “reference”로 사용)
        D_batch = self._pairwise_sq_l2(c_star)

        use_dev = (self.cur_epoch >= self.start_sls_epoch) and (self.sls is not None)
        if use_dev:
            sls = self.sls.to(D_batch.device, dtype=D_batch.dtype)
            diff = D_batch - sls[None, :, :]
            N = diff.size(1)
            dev_loss = ((diff * diff).sum(dim=(1, 2)) / float(N * N)).mean()
            total = pred_loss + self.lambda_recon * recon_loss + self.lambda_dev * dev_loss
        else:
            dev_loss = pred_loss.new_zeros(())
            total = pred_loss + self.lambda_recon * recon_loss

        return {
            "metrics": (total, pred_loss, recon_loss, dev_loss),
            "losses":  (total, pred_loss, recon_loss, dev_loss),
        }
