import numpy as np
from typing import Tuple


class AdaptiveTSConformal:
    def __init__(self, target_coverage: float = 0.85, window: int = 512, decay: float = 0.99):
        self.target_coverage = target_coverage
        self.window = window
        self.decay = decay
        self.errors = []  # store signed residuals u_real - u_pred

    def fit_partial(self, residual: float):
        self.errors.append(float(residual))
        if len(self.errors) > self.window:
            self.errors = self.errors[-self.window:]

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray, q: float) -> float:
        sorter = np.argsort(values)
        v = values[sorter]
        w = weights[sorter]
        cum = np.cumsum(w)
        cutoff = q * cum[-1]
        idx = np.searchsorted(cum, cutoff)
        idx = np.clip(idx, 0, len(v) - 1)
        return float(v[idx])

    def predict_interval(self, u_pred: float) -> Tuple[float, float]:
        if not self.errors:
            return u_pred, u_pred
        n = len(self.errors)
        values = np.asarray(self.errors, dtype=float)
        # newer samples heavier
        weights = np.asarray([self.decay ** (n - 1 - i) for i in range(n)], dtype=float)
        alpha = 1.0 - self.target_coverage
        q_hi = self._weighted_quantile(values, weights, 1.0 - alpha / 2.0)
        q_lo = self._weighted_quantile(values, weights, alpha / 2.0)
        lo = float(u_pred + q_lo)
        hi = float(u_pred + q_hi)
        return lo, hi


