from lightgbm import LGBMClassifier
def make_lgbm_movement():
    return LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=64,
                          subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                          class_weight='balanced', n_jobs=-1)
def make_lgbm_direction():
    return LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=96,
                          subsample=0.7, colsample_bytree=0.7, reg_lambda=7.0,
                          class_weight='balanced', n_jobs=-1)
