"""
Part 1C: Optuna tuning for two-stage hybrid and gating

This tunes:
- MLP hidden size and learning rate for Stage A (movement) and Stage B (direction)
- Gating thresholds (m_conf, diff)

Uses existing artifacts from Part 0/1B under /content/hybrid_workspace.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# --- Robust import of Part 1b and compatibility shim ---
import os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "part1b_two_stage_hybrid.py"),
    "/content/code/part1b_two_stage_hybrid.py",
    "/content/drive/MyDrive/Documents/code55gpt/part1b_two_stage_hybrid.py",
]
P1B_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert P1B_PATH, "Cannot find part1b_two_stage_hybrid.py"
spec = importlib.util.spec_from_file_location("p1b", P1B_PATH)
p1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1b)
print("Using Part1B from:", P1B_PATH)

# Compatibility alias
if not hasattr(p1b, "build_external_features") and hasattr(p1b, "build_context_features"):
    p1b.build_external_features = p1b.build_context_features


BASE_DIR = Path(os.environ.get('HYBRID_BASE_DIR', '/content/hybrid_workspace'))
DATASET_DIR = BASE_DIR / 'datasets' / 'chart_dataset'
EMBED_MODEL_ID = os.environ.get('EMBED_MODEL_ID', 'facebook/dinov2-base')
EMBED_DIR_OVERRIDE = os.environ.get('EMBED_DIR_OVERRIDE', '')

VAL_FRACTION = 0.2
EMBARGO_BARS = 10

# Reuse utilities from Part 1B by importing functions (simple way: exec)
def load_module(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('p1b', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def resolve_p1b_path() -> Path:
    env_path = os.environ.get('P1B_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        Path('/content/part1b_two_stage_hybrid.py'),
        Path('/content/drive/MyDrive/code/part1b_two_stage_hybrid.py'),
        Path('/Users/aidanbarber/Documents/code/part1b_two_stage_hybrid.py'),
        BASE_DIR.parent / 'code' / 'part1b_two_stage_hybrid.py',
        Path.cwd() / 'part1b_two_stage_hybrid.py',
        Path('/content/freedom44/part1b_two_stage_hybrid.py'),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Colab fallback: prompt upload
    try:
        from google.colab import files  # type: ignore
        print('Could not locate part1b_two_stage_hybrid.py. Please upload it now...')
        uploaded = files.upload()
        if 'part1b_two_stage_hybrid.py' in uploaded:
            out = Path('/content/part1b_two_stage_hybrid.py')
            with open(out, 'wb') as f:
                f.write(uploaded['part1b_two_stage_hybrid.py'])
            print('Uploaded to', out)
            return out
    except Exception:
        pass
    raise FileNotFoundError('Could not locate part1b_two_stage_hybrid.py. Set P1B_PATH env var to its absolute path.')

P1B_PATH = resolve_p1b_path()
print('Using Part1B from:', P1B_PATH)
P1B = load_module(P1B_PATH)

def prepare_data():
    meta = P1B.load_metadata()
    samples = P1B.list_samples(meta)
    n = len(samples)
    tr_idx, va_idx = P1B.split_idx(n, VAL_FRACTION, EMBARGO_BARS)
    Xd = P1B.build_dino_features(samples)
    Xt = P1B.build_ts_features(samples)
    Xy = P1B.build_yolo_pseudo_features(samples, meta)
    Xc = P1B.build_context_features(samples)
    Xe = P1B.build_external_features(samples, meta)
    Xs = P1B.build_sentiment_features(samples)
    X = np.concatenate([Xd, Xt, Xy, Xc, Xe, Xs], axis=1).astype(np.float32)
    Y = np.array([s[2] for s in samples], dtype=np.int64)
    changes = np.array([s[4] for s in samples], dtype=np.float32)
    intervals = np.array([s[6] for s in samples], dtype=np.int64)
    return X, Y, changes, intervals, tr_idx, va_idx


def objective(trial):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    X, Y, changes, intervals, tr_idx, va_idx = prepare_data()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = Y[tr_idx], Y[va_idx]
    c_va = changes[va_idx]
    iv_va = intervals[va_idx]
    X_trs, X_vas = P1B.standardize(X_tr, X_va)

    # Hyperparams
    hidden = trial.suggest_categorical('hidden', [128, 256, 384])
    lr = trial.suggest_float('lr', 3e-4, 3e-3, log=True)
    m_conf = trial.suggest_float('m_conf', 0.45, 0.70)
    diff = trial.suggest_float('diff', 0.03, 0.20)

    # Stage A
    mov = MLPClassifier(hidden_layer_sizes=(hidden,), activation='relu', alpha=1e-4,
                        batch_size=128, learning_rate_init=lr, max_iter=60, early_stopping=True)
    y_tr_m = (y_tr != 2).astype(np.int64)
    y_va_m = (y_va != 2).astype(np.int64)
    mov.fit(X_trs, y_tr_m)
    from sklearn.metrics import f1_score
    f1_mov = f1_score(y_va_m, mov.predict(X_vas), average='macro')

    # Stage B
    tr_mask = (y_tr != 2)
    va_mask = (y_va != 2)
    dirc = MLPClassifier(hidden_layer_sizes=(hidden,), activation='relu', alpha=1e-4,
                         batch_size=128, learning_rate_init=lr, max_iter=60, early_stopping=True)
    dirc.fit(X_trs[tr_mask], (y_tr[tr_mask] == 1).astype(np.int64))
    f1_dir = f1_score((y_va[va_mask]==1).astype(int), dirc.predict(X_vas[va_mask]), average='macro')

    # Backtest-like score (maximize PnL; here use pseudo Sharpe for speed)
    # Probs
    pm = mov.predict_proba(X_vas)[:,1]
    probs_dir = dirc.predict_proba(X_vas[va_mask])
    pb = np.zeros(len(pm), dtype=np.float32); pbear = np.zeros(len(pm), dtype=np.float32)
    j = 0
    for i in range(len(pm)):
        if va_mask[i]:
            pb[i] = probs_dir[j,1]; pbear[i] = probs_dir[j,0]; j += 1
    # Trades
    pnl = []
    for i in range(len(pm)):
        if pm[i] < m_conf: continue
        margin = abs(pb[i]-pbear[i])
        if margin < diff: continue
        size = max(0.0, (margin - diff) / (1.0 - diff))
        fee = 0.001
        if pb[i] > pbear[i]: pnl.append(size * (float(c_va[i]) - fee))
        else: pnl.append(size * (float(-c_va[i]) - fee))
    pnl = np.array(pnl, dtype=np.float32)
    if len(pnl) == 0: return -1.0
    score = pnl.mean() / (pnl.std() + 1e-8)  # Sharpe proxy
    # Regularize with f1s
    return score + 0.05 * (f1_mov + f1_dir)


def main():
    # Ensure optuna is available in Colab
    try:
        import optuna  # type: ignore
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna'])
        import optuna  # type: ignore
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('Best params:', study.best_params)
    out = BASE_DIR / 'hybrid_models' / 'optuna_best.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, 'w') as f:
        json.dump(study.best_params, f)
    print('Saved best params to', out)


if __name__ == '__main__':
    main()


