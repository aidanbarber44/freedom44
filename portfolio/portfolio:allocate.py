import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def corr_dist(corr):
    return ((1-corr)/2.0)**0.5

def hrp_weights(returns: pd.DataFrame):
    cov=returns.cov(); corr=returns.corr()
    dist=corr_dist(corr); link=linkage(squareform(dist.values), 'single')
    # quasi-recursive bisection
    sort_ix = dendrogram(link, no_plot=True)['leaves']
    w = pd.Series(1.0, index=cov.index)
    clusters=[cov.index[sort_ix]]
    def split(cluster):
        if len(cluster)<=1: return [cluster]
        mid=len(cluster)//2
        return [cluster[:mid], cluster[mid:]]
    stack=[cov.index[sort_ix]]
    while stack:
        c=stack.pop(0)
        if len(c)<=1: continue
        c1,c2=split(c)
        var1=float(np.dot(w[c1], np.dot(cov.loc[c1,c1], w[c1]))); var2=float(np.dot(w[c2], np.dot(cov.loc[c2,c2], w[c2])))
        alpha=1 - var1/(var1+var2)
        w[c1]*=alpha; w[c2]*=(1-alpha)
        stack.extend([c1,c2])
    return (w/w.sum()).sort_index()

def leverage_down(weights: pd.Series, risk_off_factor: float) -> pd.Series:
    factor = float(max(0.0, min(1.0, risk_off_factor)))
    w = weights * factor
    s = float(w.sum())
    if s > 1e-12:
        w = w / s
    return w
