import os
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from datasets.loader import get_train_dataloader, get_test_dataloader
from trainer import prepare_inputs
from utils.misc import mkdir
from models.oracle.score_oracle import ScorerOracleAD
from evaluation.annomaly_metrics import get_metrics


class DetectorOracleAD:
    def __init__(self, cfg, model):
        self.cfg = cfg.clone()

        # scoring용: stride=1, shuffle off, drop_last off
        self.cfg.DATA.TRAIN_STEP = 1
        self.cfg.TRAIN.SHUFFLE = False
        self.cfg.TRAIN.DROP_LAST = False
        self.cfg.TEST.SHUFFLE = False
        self.cfg.TEST.DROP_LAST = False

        self.model = deepcopy(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.scorer = ScorerOracleAD(self.cfg, self.model)

        self.train_loader = get_train_dataloader(self.cfg)
        self.test_loader = get_test_dataloader(self.cfg)

    @torch.no_grad()
    def predict(self):
        self.model.eval()

        self.test_scores = self._get_test_scores()
        self.test_labels = self._get_test_labels()

        assert len(self.test_scores) == len(self.test_labels), \
            f"score/label length mismatch: {len(self.test_scores)} vs {len(self.test_labels)}"

        # 논문 D.3: N=200 uniform bins에서 best F1 threshold
        self.threshold = self._find_best_threshold(self.test_scores, self.test_labels)
        pred = (self.test_scores >= self.threshold).astype(int)

        if self.cfg.TEST.POINT_ADJUST:
            pred_pa = self.point_adjust(pred, self.test_labels)
        else:
            pred_pa = None

        results = self.get_results(
            self.test_scores, pred_pa, self.test_labels,
            slidingWindow=self.cfg.TEST.SLIDING_WINDOW
        )
        self.last_results = results
        self.save_results(results)
        self.save_to_npy(**{
            "test_scores": self.test_scores,
            "test_labels": self.test_labels,
            "threshold": np.array(self.threshold)
        })

    @staticmethod
    def _find_best_threshold(scores: np.ndarray, labels: np.ndarray, N: int = 200) -> float:
        """논문 D.3: N=200 uniform bins에서 best F1 threshold 탐색"""
        s_min, s_max = scores.min(), scores.max()
        candidates = [s_min + i / (N - 1) * (s_max - s_min) for i in range(N)]

        best_f1, best_thr = -1.0, candidates[0]
        for tau in candidates:
            preds = (scores >= tau).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            p  = tp / (tp + fp + 1e-12)
            r  = tp / (tp + fn + 1e-12)
            f1 = 2 * p * r / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = tau

        return float(best_thr)

    @torch.no_grad()
    def _get_scores_all(self, dataloader, desc) -> np.ndarray:
        scores_all = []
        self.model.eval()
        for batch in tqdm(dataloader, desc=desc):
            inputs, _ = prepare_inputs(batch)
            scores = self.scorer.get_anomaly_scores(inputs)

            if scores.ndim == 1:
                scores_all.append(scores)
            elif scores.ndim == 2:
                scores_all.append(scores[:, -1])
            else:
                raise ValueError(f"Unexpected score shape: {scores.shape}")

        scores_all = torch.flatten(torch.cat(scores_all, dim=0))
        return scores_all.detach().cpu().numpy()

    def _get_test_scores(self):
        return self._get_scores_all(self.test_loader, "test scores")

    def _get_test_labels(self) -> np.ndarray:
        labels_all = []
        for batch in self.test_loader:
            _, labels = prepare_inputs(batch)
            labels_all.append(labels)
        labels_all = torch.flatten(torch.cat(labels_all, dim=0))
        return labels_all.detach().cpu().numpy()

    @staticmethod
    def point_adjust(pred, gt):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred

    @staticmethod
    def get_results(scores, pred_pa, labels, slidingWindow):
        results = get_metrics(
            score=scores,
            labels=labels,
            slidingWindow=slidingWindow,
            pred=None,        # threshold-independent metrics는 oracle threshold 사용
            version='default',
            thre=200
        )

        if pred_pa is not None:
            tp = np.sum((pred_pa == 1) & (labels == 1))
            fp = np.sum((pred_pa == 1) & (labels == 0))
            fn = np.sum((pred_pa == 0) & (labels == 1))
            precision_pa = tp / (tp + fp + 1e-12)
            recall_pa    = tp / (tp + fn + 1e-12)
            f1_pa = 2 * (precision_pa * recall_pa) / (precision_pa + recall_pa + 1e-12)
            results.update({
                "Precision_PA": precision_pa,
                "Recall_PA": recall_pa,
                "F1_PA": f1_pa
            })

        return results

    def save_results(self, results):
        results_string = ", ".join([f"{k}: {v:.4f}" for k, v in results.items()])
        print(results_string)
        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / "test_result.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)