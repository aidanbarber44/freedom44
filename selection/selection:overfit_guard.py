import numpy as np, math
def deflated_sharpe(sr, n, skew=0.0, kurt=3.0, n_trials=100):
    # simple practical approximation; higher is better
    psr = (sr - 0.0)/max(1e-9, math.sqrt((1 + (skew**2)/4 + (kurt-3)/6)/n))
    # deflate for multiple trials
    return psr - 2*math.sqrt(math.log(max(2,n_trials))/max(1,n))

def pbo_from_matrix(perf_matrix):
    # perf_matrix: rows = OOS evaluations; cols = strategies/configs
    # compute fraction where IS winner is below median OOS rank
    oos_ranks = (perf_matrix.argsort(axis=1)).astype(float)
    is_winner = perf_matrix.argmax(axis=1)
    below = (oos_ranks[np.arange(len(is_winner)), is_winner] < (perf_matrix.shape[1]/2)).mean()
    return 1.0 - below
