import pandas as pd
def add_funding_basis(df, funding=None, basis=None):
    # funding/basis are optional panels aligned on index; if None -> return empty
    if funding is None and basis is None:
        return pd.DataFrame(index=df.index)
    cols={}
    if funding is not None:
        cols['fund_lvl']=funding
        cols['fund_chg']=funding.diff()
    if basis is not None:
        cols['basis_lvl']=basis
        cols['basis_chg']=basis.diff()
    return pd.DataFrame(cols, index=df.index)
