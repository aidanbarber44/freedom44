import numpy as np, pandas as pd
from typing import Dict, Literal, Tuple


TP = 1
SL = 2
CENSORED = 0


def _compute_event_and_time(
    close: pd.Series,
    atr: pd.Series,
    idx: int,
    tp_mult: float,
    sl_mult: float,
    horizon_max: int,
    mode: Literal["bars", "vol_scaled"],
) -> Tuple[int, int, float]:
    """
    Scan forward from idx until TP/SL/timeout.
    Returns (risk, time_to_event_bars, time_scale_units)
    - risk in {TP, SL, CENSORED}
    - time_to_event_bars in 1..horizon_max (or horizon_max if censored)
    - time_scale_units is bars if mode==bars else cumulative ATR units until event/timeout
    """
    p0 = float(close.iloc[idx])
    a0 = float(atr.iloc[idx] or 0.0)
    ub = p0 + tp_mult * a0
    lb = p0 - sl_mult * a0

    forward_close = close.iloc[idx + 1 : idx + 1 + horizon_max]
    forward_atr = atr.iloc[idx + 1 : idx + 1 + horizon_max]

    cum_units = 0.0
    risk, t_bars = CENSORED, horizon_max
    for step, (p, a) in enumerate(zip(forward_close.values, forward_atr.values), start=1):
        if mode == "vol_scaled":
            cum_units += float(a or 0.0)
        if p >= ub:
            risk, t_bars = TP, step
            break
        if p <= lb:
            risk, t_bars = SL, step
            break
    if mode == "bars":
        cum_units = float(t_bars)
    return risk, t_bars, cum_units


def build_survival_targets(
    df: pd.DataFrame,
    tp_mult: float,
    sl_mult: float,
    horizon_max: int,
    mode: Literal["bars", "vol_scaled"] = "bars",
    bins: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Build discrete-time competing-risks survival targets from triple-barrier rules.

    Inputs:
      df: must contain columns ['close','atr'] indexed by time.
      tp_mult/sl_mult: ATR-scaled barriers.
      horizon_max: max look-ahead bars to resolve event before censor.
      mode: 'bars' or 'vol_scaled' to determine discretization scale.
      bins: number of discrete time bins K.

    Outputs dict with arrays aligned to df index (excluding last horizon_max rows):
      - event_risk: int in {TP(1), SL(2), CENSORED(0)}
      - event_time_bin: int in [1..K] (K used for censored tail)
      - is_censored: 0/1
      - K: number of bins
    """
    assert "close" in df.columns and "atr" in df.columns, "df must have close and atr"
    n = len(df)
    if n <= horizon_max:
        raise ValueError("Not enough rows for horizon_max")

    close = df["close"].astype(float)
    atr = df["atr"].astype(float).ffill().fillna(0.0)

    # Pre-compute scale for vol_scaled binning
    # Max cumulative units for timeout path
    if mode == "bars":
        total_units_timeout = float(horizon_max)
    else:
        rolling_units = atr.rolling(window=horizon_max, min_periods=1).sum()
        total_units_timeout = float(rolling_units.max() or 1.0)
        total_units_timeout = max(total_units_timeout, 1e-6)

    risks = []
    t_bins = []
    cens = []

    for i in range(n - horizon_max):
        risk, t_bars, units = _compute_event_and_time(
            close, atr, i, tp_mult, sl_mult, horizon_max, mode
        )
        # Map time to discrete bin 1..K
        if mode == "bars":
            frac = float(t_bars) / float(horizon_max)
        else:
            # normalize by max units for censor path; guard tiny denom
            frac = float(units) / float(total_units_timeout)
        frac = min(max(frac, 0.0), 1.0)
        k = int(np.ceil(frac * bins))
        k = max(1, min(bins, k))
        risks.append(risk)
        t_bins.append(k)
        cens.append(1 if risk == CENSORED else 0)

    out = {
        "event_risk": np.asarray(risks, dtype=np.int64),
        "event_time_bin": np.asarray(t_bins, dtype=np.int64),
        "is_censored": np.asarray(cens, dtype=np.int64),
        "K": int(bins),
    }

    # basic sanity
    assert out["event_time_bin"].min() >= 1 and out["event_time_bin"].max() <= bins
    return out


# Quick unit checks (lightweight, import-safe)
def _unit_test_basic():
    idx = pd.date_range("2024-01-01", periods=200, freq="H")
    close = pd.Series(np.cumsum(np.random.randn(200) * 0.1) + 100.0, index=idx)
    atr = close.rolling(14).std().fillna(close.std() * 0.1)
    df = pd.DataFrame({"close": close, "atr": atr})
    t = build_survival_targets(df, tp_mult=2.0, sl_mult=1.0, horizon_max=12, mode="bars", bins=16)
    assert len(t["event_risk"]) == len(df) - 12
    assert (t["event_time_bin"] >= 1).all() and (t["event_time_bin"] <= 16).all()


if __name__ == "__main__":
    _unit_test_basic()


