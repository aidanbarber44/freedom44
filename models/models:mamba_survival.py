import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Optional

HAS_MAMBA=False
try:
    from mamba_ssm import Mamba
    HAS_MAMBA=True
except Exception:
    HAS_MAMBA=False


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, encoder: str = "mamba"):
        super().__init__()
        self.encoder=encoder
        if encoder == "mamba" and HAS_MAMBA:
            layers=[]
            d_model=hidden_size
            layers.append(nn.Linear(input_dim, d_model))
            for _ in range(num_layers):
                layers.append(Mamba(d_model))
                layers.append(nn.Dropout(dropout))
            self.net=nn.Sequential(*layers)
            self.out_dim=d_model
        else:
            self.rnn=nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.out_dim=hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, F]
        if getattr(self, 'encoder', 'gru') == 'mamba' and HAS_MAMBA:
            h=self.net(x)
            return h[:, -1, :]
        out, h_n = self.rnn(x)
        return out[:, -1, :]


class SurvivalMamba(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, K: int, dropout: float = 0.1, encoder: str = "mamba", use_movement_head: bool = True):
        super().__init__()
        self.enc = Encoder(input_dim, hidden_size, num_layers, dropout, encoder)
        self.K=K
        self.use_movement_head = use_movement_head
        self.h_tp = nn.Linear(self.enc.out_dim, K)
        self.h_sl = nn.Linear(self.enc.out_dim, K)
        if use_movement_head:
            self.h_move = nn.Linear(self.enc.out_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.enc(x)
        hazards_tp = torch.sigmoid(self.h_tp(h))
        hazards_sl = torch.sigmoid(self.h_sl(h))
        out = {"hazards_tp": hazards_tp, "hazards_sl": hazards_sl}
        if self.use_movement_head:
            out["p_move"] = torch.sigmoid(self.h_move(h)).squeeze(-1)
        return out


def cif_from_hazards(h_tp: torch.Tensor, h_sl: torch.Tensor) -> Dict[str, torch.Tensor]:
    # hazards: [B,K]
    B, K = h_tp.shape
    s = torch.ones(B, K, device=h_tp.device)
    cif_tp = torch.zeros(B, K, device=h_tp.device)
    cif_sl = torch.zeros(B, K, device=h_tp.device)
    surv = torch.ones(B, device=h_tp.device)
    for k in range(K):
        total_h = h_tp[:, k] + h_sl[:, k]
        cif_tp[:, k] = surv * h_tp[:, k]
        cif_sl[:, k] = surv * h_sl[:, k]
        surv = surv * (1.0 - total_h)
    return {"tp": cif_tp, "sl": cif_sl}


def p_tp_before_sl(h_tp: torch.Tensor, h_sl: torch.Tensor) -> torch.Tensor:
    B, K = h_tp.shape
    surv = torch.ones(B, device=h_tp.device)
    p = torch.zeros(B, device=h_tp.device)
    for k in range(K):
        p = p + surv * h_tp[:, k]
        surv = surv * (1.0 - h_tp[:, k] - h_sl[:, k])
    return p


def deephit_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], lambda_move: float = 0.05, p_move: Optional[torch.Tensor] = None, use_movement_head: bool = False) -> torch.Tensor:
    # targets: risk in {0:cens,1:TP,2:SL}, time_bin in [1..K]
    h_tp, h_sl = outputs["hazards_tp"], outputs["hazards_sl"]
    B, K = h_tp.shape
    risk = targets["risk"]  # [B]
    tbin = targets["time_bin"]  # [B]

    # likelihood per sample
    loglik = torch.zeros(B, device=h_tp.device)
    for b in range(B):
        k = int(tbin[b].item()) - 1
        surv = torch.tensor(1.0, device=h_tp.device)
        for j in range(k):
            surv = surv * (1.0 - h_tp[b, j] - h_sl[b, j])
        if risk[b].item() == 1:  # TP
            loglik[b] = torch.log(surv * h_tp[b, k] + 1e-12)
        elif risk[b].item() == 2:  # SL
            loglik[b] = torch.log(surv * h_sl[b, k] + 1e-12)
        else:
            # censored: survive through k
            for j in range(k, K):
                surv = surv * (1.0 - h_tp[b, j] - h_sl[b, j])
            loglik[b] = torch.log(surv + 1e-12)

    loss = -loglik.mean()
    if use_movement_head and p_move is not None and "p_move" in outputs:
        move_target = (risk != 0).float()
        loss = loss + lambda_move * F.binary_cross_entropy(outputs["p_move"], move_target)
    return loss


def expected_utility(
    h_tp: torch.Tensor,
    h_sl: torch.Tensor,
    tp_pl: float,
    sl_pl: float,
    fees: float,
    side: str = "long",
) -> torch.Tensor:
    p_tp = p_tp_before_sl(h_tp, h_sl)
    p_sl = 1.0 - p_tp  # approximation
    if side == "long":
        util = p_tp * tp_pl - p_sl * sl_pl - fees
    else:
        util = p_tp * tp_pl - p_sl * sl_pl - fees
    return util


