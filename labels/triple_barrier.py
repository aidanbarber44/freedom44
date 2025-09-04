import numpy as np, pandas as pd, logging

def _compute_atr(close: pd.Series, high=None, low=None, window: int = 14) -> pd.Series:
    """
    Compute ATR from high/low/close if available; otherwise use a proxy
    (rolling std of returns × price). Returns a Series aligned to `close`.
    """
    if high is not None and low is not None:
        try:
            tr1 = (high - low).abs()
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window, min_periods=window//2).mean()
            return atr
        except Exception:
            logging.exception("High/low ATR computation failed; falling back to proxy ATR")

    # Proxy: price-vol
    ret_std = close.pct_change().rolling(window, min_periods=window//2).std().shift(1)
    atr_proxy = (ret_std * close)
    return atr_proxy


def generate_events(close: pd.Series, ub_mult=2.5, lb_mult=0.75, t_max=12, atr=None, percent_mode: bool = False):
    # Use ATR(14) or proxy as scale by default; optional percent_mode available
    events=[]
    for i in range(len(close)-t_max):
        p0=close.iloc[i]
        if percent_mode:
            ub = p0 * (1 + ub_mult)
            lb = p0 * (1 - lb_mult)
        else:
            if atr is None:
                atr_series = _compute_atr(close)
            else:
                atr_series = atr
            # Guard early-window NaNs and ensure numeric
            atr_series = atr_series.bfill().fillna(0.0)
            scale = float(atr_series.iloc[i])
            ub = p0 + ub_mult * scale
            lb = p0 - lb_mult * scale
        # resolve by walk-forward within t_max: hit ub -> +1, lb -> -1, none -> 0
        w=close.iloc[i+1:i+1+t_max]
        label=0
        if (w>=ub).any(): label=1
        elif (w<=lb).any(): label=-1
        events.append((close.index[i], label))
    return pd.DataFrame(events, columns=['ts','label']).set_index('ts')

def meta_labels(primary_prob, features_df):
    # Primary Stage-B signal strength -> meta y: take(1)/skip(0)
    edge = (primary_prob[:,1]-primary_prob[:,0])  # bull - bear
    y_meta = (abs(edge) > np.quantile(abs(edge), 0.6)).astype(int)  # simple baseline
    sw = 1.0 + 1.0*(abs(edge) > np.quantile(abs(edge), 0.8))
    return y_meta, sw
