import numpy as np, pandas as pd

def _safe_div(a, b):
    """
    Safely divide two Pandas Series, returning a Pandas Series.
    The original function returned a numpy array, causing an error.
    """
    # Ensure inputs are Pandas Series to preserve index
    a_series = pd.Series(a)
    b_series = pd.Series(b)
    # Perform the division using np.where for safety, then wrap it back into a Pandas Series
    result_array = np.where(b_series == 0, 0.0, a_series / b_series)
    return pd.Series(result_array, index=a_series.index)

def add_microstructure(df, price='close', high='high', low='low', vol='volume'):
    out = pd.DataFrame(index=df.index)
    r1 = df[price].pct_change()
    out['ret1']=r1
    out['ret5']=df[price].pct_change(5)
    out['ret20']=df[price].pct_change(20)
    
    out['rv5']=(r1.rolling(5).std()*np.sqrt(1440)).fillna(0)
    out['rv20']=(r1.rolling(20).std()*np.sqrt(1440)).fillna(0)
    
    # This line now correctly receives a Pandas Series from _safe_div
    hl = _safe_div(df[high]-df[low], df[price])
    out['range5']=hl.rolling(5).mean()
    out['range20']=hl.rolling(20).mean()
    
    up = (df[price].diff()>0).astype(int)
    down=(df[price].diff()<0).astype(int)
    ofi = (up*df[vol]-down*df[vol])
    out['ofi1']=ofi
    out['ofi5']=ofi.rolling(5).sum()
    out['ofi20']=ofi.rolling(20).sum()
    
    tr = np.maximum(df[high]-df[low], np.maximum(abs(df[high]-df[price].shift()), abs(df[low]-df[price].shift())))
    atr20 = tr.rolling(20).mean()
    ma20=df[price].rolling(20).mean()
    std20=df[price].rolling(20).std()
    
    bbw = _safe_div(2*std20, ma20)
    out['squeeze']=_safe_div(bbw,(atr20/df[price]))
    
    ema12=df[price].ewm(span=12,adjust=False).mean()
    ema26=df[price].ewm(span=26,adjust=False).mean()
    macd=ema12-ema26
    sig=macd.ewm(span=9,adjust=False).mean()
    out['macd']=macd
    out['macd_sig']=sig
    out['macd_hist']=macd-sig
    
    return out
