import os, random, numpy as np
def set_all_seeds(seed:int=42):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    np.random.seed(seed)
