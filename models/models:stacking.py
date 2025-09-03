import numpy as np
from sklearn.linear_model import LogisticRegression
class ProbabilityStacker:
    def __init__(self, C=1.0):
        self.clf=LogisticRegression(C=C, max_iter=300, class_weight='balanced')
    def fit(self, P, y): self.clf.fit(P,y); return self
    def predict_proba(self, P): return self.clf.predict_proba(P)
