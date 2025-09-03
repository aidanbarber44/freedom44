import pandas as pd
def add_har_rv(ret: pd.Series):
    absr = ret.abs()
    return pd.DataFrame({'har_d':absr.rolling(1).mean(),
                         'har_w':absr.rolling(5).mean(),
                         'har_m':absr.rolling(22).mean()}, index=ret.index)
