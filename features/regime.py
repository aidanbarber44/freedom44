import numpy as np, pandas as pd
def add_regime(close: pd.Series):
    z20=(close-close.rolling(20).mean())/(close.rolling(20).std()+1e-9)
    slope5=close.diff(5)
    return pd.DataFrame({'z20':z20,'slope5':slope5,
                         'regime_trend':((z20>0)&(slope5>0)).astype(int)-((z20<0)&(slope5<0)).astype(int)},
                         index=close.index)
