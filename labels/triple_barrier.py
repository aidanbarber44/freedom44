import numpy as np, pandas as pd
def generate_events(close: pd.Series, ub_mult=2.5, lb_mult=0.75, t_max=12, atr=None):
    # Use ATR(14) or rolling TR as scale if provided; else pct thresholds
    events=[]
    for i in range(len(close)-t_max):
        p0=close.iloc[i]
        ub=p0*(1+0.0) if atr is None else p0+ub_mult*(atr.iloc[i] or 0)
        lb=p0*(1-0.0) if atr is None else p0-lb_mult*(atr.iloc[i] or 0)
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
