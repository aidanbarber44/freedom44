import numpy as np
from sklearn.model_selection import KFold
def purged_time_splits(n:int, k:int, embargo:int):
    idx=np.arange(n)
    kf=KFold(n_splits=k, shuffle=False)
    for tr,vl in kf.split(idx):
        # apply embargo: drop last 'embargo' of train near val edges
        vl_start, vl_end = vl[0], vl[-1]
        tr_mask=(idx <= vl_start-embargo) | (idx >= vl_end+embargo)
        yield idx[tr_mask], vl
