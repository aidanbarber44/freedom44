import numpy as np
class RollingConformal:
    def __init__(self, alpha=0.2, window=1000):
        self.alpha=alpha; self.window=window; self.res=[]
    def update(self, error):
        self.res.append(abs(error))
        if len(self.res)>self.window: self.res=self.res[-self.window:]
    def q(self):
        if not self.res: return 0.0
        return float(np.quantile(self.res, 1.0-self.alpha))
