import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer import Trainer, prepare_inputs
from models.oracle.detector import DetectorOracleAD


class OracleADTrainer(Trainer):

    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.start_sls_epoch = int(self.cfg.ORACLEAD.SLS.START_EPOCH)
        self.lambda_recon = float(self.cfg.ORACLEAD.LAMBDA_RECON)
        self.lambda_dev = float(self.cfg.ORACLEAD.LAMBDA_DEV)
        self.use_squared_l2 = bool(getattr(self.cfg.ORACLEAD.SLS, "USE_SQUARED_L2", False))

        self.sls = None
        self._reset_sls_accum()

        sls_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "sls_latest.pt")
        if os.path.isfile(sls_path):
            self.sls = torch.load(sls_path, map_location="cpu")
            print(f"[OracleADTrainer] Loaded SLS from {sls_path} (shape={tuple(self.sls.shape)})")

    # ------------------------------------------------------------------ #
    #  train() override: TensorBoard + eval period마다 test 실행           #
    # ------------------------------------------------------------------ #
    def train(self):
        metric_best = self.cfg.TRAIN.METRIC_BEST
        save_path = Path(self.cfg.RESULT_DIR)

        tb_dir = save_path / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard logs: {tb_dir}")

        # Config에서 EVAL_ENABLE 여부 확인 (기본값은 False로 안전하게 처리)
        eval_enable = getattr(self.cfg.TRAIN, "EVAL_ENABLE", False)

        for cur_epoch in tqdm(range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH)):
            train_losses = self.train_epoch()

            # train loss 로깅
            if train_losses and self.writer:
                for name, val in train_losses.items():
                    self.writer.add_scalar(f"Train/{name}", val, cur_epoch)

            # 1. EVAL_ENABLE이 True일 때만 기존의 검증(Validation) 로직 수행
            if eval_enable and self._is_eval_epoch(cur_epoch):
                tracking_meter, val_losses = self.eval_epoch()

                # val loss 로깅
                if self.writer:
                    self.writer.add_scalar(f"Val/{tracking_meter.name}",
                                           tracking_meter.avg, cur_epoch)
                    for name, val in val_losses.items():
                        self.writer.add_scalar(f"Val/{name}", val, cur_epoch)

                is_best = self._check_improvement(tracking_meter.avg, metric_best)
                if is_best:
                    with open(save_path / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    print(f"[current best] Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    metric_best = tracking_meter.avg

            # 2. EVAL_ENABLE이 False라면, 모델을 최신 상태로 덮어쓰며 저장 (논문 방식)
            elif not eval_enable:
                # Trainer 구조에 맞게 최신 모델을 저장. (save_best_model을 그대로 호출하거나 별도 함수 사용)
                self.save_best_model() 

            # 3. Test 로깅: 학습 중간에 Test 점수를 텐서보드로 보고 싶을 때만 수행
            if self._is_eval_epoch(cur_epoch):
                self._log_test_metrics(cur_epoch)

            self.cur_epoch += 1

        if self.writer:
            self.writer.close()

    def _log_test_metrics(self, cur_epoch):
        print(f"[Epoch {cur_epoch}] Running test...")
        predictor = DetectorOracleAD(self.cfg, self.model)
        predictor.predict()

        if self.writer and hasattr(predictor, 'last_results'):
            for name, val in predictor.last_results.items():
                self.writer.add_scalar(f"Test/{name}", float(val), cur_epoch)
            print(f"[Epoch {cur_epoch}] Test metrics logged to TensorBoard")

    # ------------------------------------------------------------------ #
    #  SLS 관련                                                            #
    # ------------------------------------------------------------------ #
    def _reset_sls_accum(self):
        self.sls_sum = None
        self.sls_count = 0

    @staticmethod
    def _pairwise_sq_l2(c_star: torch.Tensor) -> torch.Tensor:
        x2 = (c_star * c_star).sum(dim=-1, keepdim=True)
        prod = torch.matmul(c_star, c_star.transpose(1, 2))
        dist2 = x2 + x2.transpose(1, 2) - 2.0 * prod
        return dist2.clamp_min_(0.0)

    def _pairwise_l2(self, c_star: torch.Tensor) -> torch.Tensor:
        return (self._pairwise_sq_l2(c_star) + 1e-12).sqrt()

    def _accumulate_sls(self, D_batch: torch.Tensor):
        with torch.no_grad():
            D_sum = D_batch.detach().sum(dim=0)
            self.sls_sum = D_sum if self.sls_sum is None else (self.sls_sum + D_sum)
            self.sls_count += int(D_batch.size(0))

    def _finalize_and_save_sls(self):
        if self.sls_sum is None or self.sls_count == 0:
            print("[OracleADTrainer] Warning: No D accumulated. SLS not updated.")
            return

        sls = (self.sls_sum / float(self.sls_count)).cpu()
        self.sls = sls

        if bool(getattr(self.cfg.ORACLEAD.SLS, "SAVE_EVERY_EPOCH", True)):
            ckpt_dir = self.cfg.TRAIN.CHECKPOINT_DIR
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(sls, os.path.join(ckpt_dir, f"sls_epoch_{self.cur_epoch}.pt"))
            torch.save(sls, os.path.join(ckpt_dir, "sls_latest.pt"))

    # ------------------------------------------------------------------ #
    #  train / eval                                                        #
    # ------------------------------------------------------------------ #
    def _unwrap_x(self, inputs):
        inputs = prepare_inputs(inputs)
        if isinstance(inputs, (tuple, list)):
            return inputs[0]
        return inputs

    def train_epoch(self):
        self.use_dev_loss = (self.cur_epoch >= self.start_sls_epoch) and (self.sls is not None)
        self._reset_sls_accum()
        train_losses = super().train_epoch()
        self._finalize_and_save_sls()
        return train_losses

    def _compute_pred_recon_loss(self, out: dict):
        x_hat_next = out["x_hat_next"]
        y_next = out["y_next"]
        if x_hat_next.dim() == 3 and x_hat_next.size(-1) == 1:
            x_hat_next = x_hat_next.squeeze(-1)

        x_hat_past = out["x_hat_past"]
        x_past_true = out["x_past_true"]
        if x_hat_past.dim() == 4 and x_hat_past.size(-1) == 1:
            x_hat_past = x_hat_past.squeeze(-1)

        pred_loss  = (x_hat_next - y_next).pow(2).sum(dim=-1).sqrt().mean()
        recon_loss = (x_hat_past - x_past_true).pow(2).sum(dim=(-1,-2)).sqrt().mean()
        return pred_loss, recon_loss, x_hat_next, x_hat_past

    def train_step(self, inputs):
        x = self._unwrap_x(inputs)
        out = self.model(x)

        pred_loss, recon_loss, _, _ = self._compute_pred_recon_loss(out)
        c_star = out["c_star"]

        D_batch = self._pairwise_sq_l2(c_star) if self.use_squared_l2 else self._pairwise_l2(c_star)
        self._accumulate_sls(D_batch)

        if self.use_dev_loss:
            sls = self.sls.to(D_batch.device, dtype=D_batch.dtype)
            diff = D_batch - sls[None, :, :]
            N = diff.size(1)
            dev_loss = (diff.pow(2).sum(dim=(1, 2)) / float(N * N)).mean()
            total = pred_loss + self.lambda_recon * recon_loss + self.lambda_dev * dev_loss
        else:
            dev_loss = pred_loss.new_zeros(())
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

        pred_loss, recon_loss, _, _ = self._compute_pred_recon_loss(out)
        c_star = out["c_star"]

        D_batch = self._pairwise_sq_l2(c_star) if self.use_squared_l2 else self._pairwise_l2(c_star)

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