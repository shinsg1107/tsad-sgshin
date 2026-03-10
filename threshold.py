from typing import Optional
from functools import cached_property

import numpy as np
import sklearn.metrics as metrics


class Thresholder:
    def __init__(
            self,
            cfg,
            test_scores: np.ndarray,
            test_labels: np.ndarray,
            train_scores: Optional[np.ndarray] = None,
            val_scores: Optional[np.ndarray] = None
    ):
        self.cfg = cfg
        self.test_scores = test_scores
        self.test_labels = test_labels
        if isinstance(train_scores, np.ndarray):
            self.train_scores = train_scores
        if isinstance(val_scores, np.ndarray):
            self.val_scores = val_scores

        self.threshold_type = cfg.TEST.THRESHOLD.TYPE
        assert self.threshold_type in ('ratio', 'best_f1')

    @cached_property
    def threshold_best_f1(self):
        N = 200
        s_min, s_max = self.test_scores.min(), self.test_scores.max()
        candidates = [s_min + i / (N - 1) * (s_max - s_min) for i in range(N)]

        best_f1 = -1
        best_threshold = candidates[0]
        
        for tau in candidates:
            preds = (self.test_scores >= tau).astype(int)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                self.test_labels, preds, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = tau
        
        return best_threshold

    @cached_property
    def threshold_ratio(self):
        # adapted from: https://github.com/thuml/Anomaly-Transformer/solver.py
        assert hasattr(self, 'val_scores')
        self.ratio = self.cfg.TEST.THRESHOLD.ANOMALY_RATIO
        threshold_ratio = np.percentile(self.val_score, 100 - self.ratio)

        return threshold_ratio

    @property
    def threshold(self):
        if self.threshold_type == 'ratio':
            return self.threshold_ratio
        elif self.threshold_type == 'best_f1':
            return self.threshold_best_f1
