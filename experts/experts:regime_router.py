import numpy as np
from sklearn.linear_model import LogisticRegression
class Router:
    def __init__(self): self.clf=LogisticRegression(max_iter=200)
    def fit(self, X, y): self.clf.fit(X,y); return self
    def predict_proba(self, X): return self.clf.predict_proba(X)
