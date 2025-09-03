import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any

try:
    import signatory  # type: ignore
    HAS_SIGNATORY=True
except Exception:
    HAS_SIGNATORY=False


def compute_logsignature(tensor_window: torch.Tensor, depth: int) -> torch.Tensor:
    if not HAS_SIGNATORY:
        return torch.empty((tensor_window.shape[0], 0), dtype=tensor_window.dtype, device=tensor_window.device)
    # tensor_window: [W, F]
    path = tensor_window.unsqueeze(0)
    return signatory.logsignature(path, depth=depth).squeeze(0)


class SequenceSurvivalDataset(Dataset):
    def __init__(
        self,
        features_df: pd.DataFrame,
        targets: Dict[str, np.ndarray],
        window: int,
        signature_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        self.X = features_df.values.astype(np.float32)
        self.index = features_df.index
        self.window = int(window)
        self.tgt = targets
        self.device = device
        self.sig_enabled = bool(signature_cfg.get("enabled", False)) if signature_cfg else False
        self.sig_depth = int(signature_cfg.get("depth", 3)) if signature_cfg else 3
        self.sig_window = int(signature_cfg.get("window", window)) if signature_cfg else window
        self.num_rows = len(features_df)
        # Align length: targets computed up to n - horizon
        self.valid_len = len(self.tgt["event_risk"])  # excludes last horizon_max rows

        self.num_features = self.X.shape[1]

    def __len__(self):
        return max(0, self.valid_len - self.window + 1)

    def __getitem__(self, i: int):
        j = i + self.window
        x_seq = torch.from_numpy(self.X[i:j])  # [W, F]
        if self.sig_enabled and self.sig_window <= self.window:
            sig_start = j - self.sig_window
            sig_feats = compute_logsignature(x_seq[sig_start - i :], depth=self.sig_depth)
            if sig_feats.numel() > 0:
                # append to last step features
                last = x_seq[-1]
                x_seq[-1] = torch.cat([last, sig_feats], dim=-1)
        meta = {"start_idx": i, "end_idx": j, "ts": self.index[j - 1]}
        y = {
            "risk": int(self.tgt["event_risk"][j - 1]),
            "time_bin": int(self.tgt["event_time_bin"][j - 1]),
            "is_censored": int(self.tgt["is_censored"][j - 1]),
            "K": int(self.tgt["K"]),
        }
        return x_seq, meta, y


