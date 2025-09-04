"""
Part 1: Hybrid training and backtest (no regeneration)

What it does quickly on existing data:
- Loads metadata and precomputed DINO embeddings
- Builds YOLO-like features from pseudo-labels stored in metadata (fast; no YOLO inference)
- Builds compact time-series summary features from saved window pickles
- Temporal split with optional embargo
- Trains a calibrated Logistic Regression on concatenated features (DINO + YOLO + TS)
- Runs a gated backtest on the validation tail over avg_change (long/short/hold)

Outputs:
- Prints feature shapes, class distribution, F1/Acc
- Sweeps confidence/margin thresholds; reports best PnL, Sharpe, WinRate, Max Drawdown
- Saves scaler and model to workspace for reuse, hello world!

Notes:
- Designed to run in Colab quickly using existing /content/hybrid_workspace artifacts from Part 0
"""




import os
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

import numpy as np
import pandas as pd

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

# === NEW IMPORTS (idempotent) ===
try:
    import yaml  # type: ignore
except Exception:  # minimal fallback if YAML unavailable
    yaml = None  # type: ignore
try:
    from utils.seed import set_all_seeds  # type: ignore
except Exception:
    def set_all_seeds(seed: int) -> None:
        # graceful no-op if helper not present
        import random
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        random.seed(seed)
        np.random.seed(seed)
try:
    from cv.purged import purged_time_splits  # type: ignore
except Exception:
    purged_time_splits = None  # type: ignore
try:
    from features.microstructure import add_microstructure  # type: ignore
    from features.volatility import add_har_rv  # type: ignore
    from features.regime import add_regime  # type: ignore
    from features.funding_basis import add_funding_basis  # type: ignore
except Exception:
    # provide light stubs so script remains runnable without the optional modules
    def add_microstructure(df):
        return pd.DataFrame(index=getattr(df, 'index', None))
    def add_har_rv(rets):
        return pd.DataFrame(index=getattr(rets, 'index', None))
    def add_regime(close):
        return pd.DataFrame(index=getattr(close, 'index', None))
    def add_funding_basis(ohlcv, funding=None, basis=None):
        return pd.DataFrame(index=getattr(ohlcv, 'index', None))
try:
    from models.gbm import make_lgbm_movement, make_lgbm_direction  # type: ignore
except Exception:
    def make_lgbm_movement():
        # lightweight fallback to LR if LightGBM wrapper missing
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)
    def make_lgbm_direction():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)
try:
    from models.stacking import ProbabilityStacker  # type: ignore
except Exception:
    ProbabilityStacker = None  # type: ignore

# === CONFIG LOADER (idempotent) ===
def load_config(path: str = "conf/experiment.yaml"):
    if path and os.path.exists(path) and yaml is not None:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            logging.error(f"Error loading config from {path}")
            return {
                "execution": {
                    "tp_sl": {"atr_mult_tp": 2.5, "atr_mult_sl": 0.75},
                    "kelly_scale": 0.35, "size_cap": 0.05,
                },
                "run": {"mode": "research", "stressed": True},
            }
    logging.warning(f"Config not found at {path}, using default.")
    return {
        "execution": {
            "tp_sl": {"atr_mult_tp": 2.5, "atr_mult_sl": 0.75},
            "kelly_scale": 0.35, "size_cap": 0.05,
        },
        "run": {"mode": "research", "stressed": True},
    }

CONFIG = load_config()

# === SEED (must happen before any training) ===
set_all_seeds(42)

# === ATR helper ===
def compute_atr14(ohlcv: pd.DataFrame) -> pd.Series:
    high = ohlcv.get('high')
    low = ohlcv.get('low')
    close = ohlcv.get('close')
    if high is None or low is None or close is None:
        return pd.Series(index=getattr(ohlcv, 'index', None), dtype=float)
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    return tr.rolling(14).mean()




# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(os.environ.get('HYBRID_BASE_DIR', '/content/hybrid_workspace'))
DATASET_DIR = BASE_DIR / 'datasets' / 'chart_dataset'
EMBED_DIR = DATASET_DIR / 'embeddings_dino'
# Allow DINOv3 or custom model embeddings directory via env
EMBED_MODEL_ID = os.environ.get('EMBED_MODEL_ID', 'facebook/dinov2-base')
EMBED_DIR_OVERRIDE = os.environ.get('EMBED_DIR_OVERRIDE', '')
EMBED_DIR_V2 = DATASET_DIR / 'embeddings_dino'
EMBED_DIR_V3 = DATASET_DIR / 'embeddings_dinov3'
if EMBED_DIR_OVERRIDE:
    EMBED_DIR = Path(EMBED_DIR_OVERRIDE)
elif 'dinov3' in EMBED_MODEL_ID.lower():
    EMBED_DIR = EMBED_DIR_V3

EMBARGO_BARS = 10  # prevent leakage near split boundary (applies in 4H bars approx)
VAL_FRACTION = 0.2


# ----------------------------
# Helpers
# ----------------------------
def load_metadata() -> Dict[str, dict]:
    meta_path = DATASET_DIR / 'metadata.json'
    assert meta_path.exists(), f"metadata.json missing at {meta_path}"
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata from {meta_path}: {e}")
        return {}


def list_samples(meta: Dict[str, dict]) -> List[Tuple[datetime, str, int, str, float, str, int]]:
    label_map = {'bearish': 0, 'bullish': 1, 'neutral': 2}
    samples: List[Tuple[datetime, str, int, str, float, str, int]] = []
    for fname, m in meta.items():
        lab = m.get('label')
        ts_iso = m.get('timestamp')
        wfile = m.get('window_file')
        avg_change = float(m.get('avg_change', 0.0))
        asset = m.get('asset', 'UNK')
        interval_min = int(m.get('interval_min', 0))
        if lab not in label_map:
            continue
        # Resolve image path
        img_path = DATASET_DIR / lab / fname
        if not img_path.exists():
            continue
        try:
            ts = datetime.fromisoformat(ts_iso)
        except Exception:
            logging.warning(f"Skipping sample with invalid timestamp: {ts_iso}")
            continue
        samples.append((ts, str(img_path), label_map[lab], wfile, avg_change, asset, interval_min))
    samples.sort(key=lambda x: x[0])
    return samples


def train_val_split(samples: List[Tuple[datetime, str, int, str, float, str, int]], val_fraction: float, embargo: int) -> Tuple[List[int], List[int]]:
    n = len(samples)
    split = int((1.0 - val_fraction) * n)
    # Embargo around split
    train_idx = list(range(0, max(0, split - embargo)))
    val_idx = list(range(min(n, split + embargo), n))
    return train_idx, val_idx


def load_dino_emb(path: Path) -> np.ndarray:
    try:
        v = np.asarray(np.load(path, allow_pickle=True))  # fallback if saved differently
        if v.ndim == 1:
            return v.astype(np.float32)
    except Exception as e:
        logging.warning(f"Error loading DINO embedding from {path}: {e}")
    # Standard torch .pt
    try:
        import torch
        t = torch.load(str(path))
        return (t if isinstance(t, np.ndarray) else t.cpu().numpy()).astype(np.float32)
    except Exception as e:
        logging.error(f"Error loading DINO embedding from {path}: {e}")
    return np.zeros(768, dtype=np.float32) # Return a default zero array


def build_dino_features(samples: List[Tuple[datetime, str, int, str, float, str, int]]) -> np.ndarray:
    feats: List[np.ndarray] = []
    for s in samples:
        img_path = s[1]
        stem = Path(img_path).stem
        pt = EMBED_DIR / f"{stem}.pt"
        if not pt.exists():
            logging.warning(f"DINO embedding not found for {img_path}, using zeros.")
            feats.append(np.zeros(768, dtype=np.float32))
            continue
        feats.append(load_dino_emb(pt))
    X = np.vstack(feats).astype(np.float32)
    return X


def build_yolo_pseudo_features(samples: List[Tuple[datetime, str, int, str, float, str, int]], meta: Dict[str, dict]) -> np.ndarray:
    # names: 8 classes, use count + area sum per class => 16-dim
    names = ['fvg_bull','fvg_bear','ob_bull','ob_bear','bos_bull','bos_bear','choch_bull','choch_bear']
    feats: List[np.ndarray] = []
    for s in samples:
        img_path = s[1]
        fn = Path(img_path).name
        m = meta.get(fn, {})
        boxes = m.get('yolo_pseudo_boxes', [])
        vec = np.zeros(2*len(names), dtype=np.float32)
        for b in boxes:
            try:
                cls_id, xc, yc, w, h = b
                if 0 <= cls_id < len(names):
                    vec[cls_id] += 1.0
                    vec[len(names)+cls_id] += float(w) * float(h)
            except Exception:
                logging.warning(f"Skipping invalid YOLO pseudo-box for {img_path}: {b}")
                continue
        feats.append(vec)
    return np.vstack(feats).astype(np.float32)


def build_external_features(samples: List[Tuple[datetime, str, int, str, float, str, int]], meta: Dict[str, dict]) -> np.ndarray:
    # Discover all keys
    key_set = set()
    for s in samples:
        fn = Path(s[1]).name
        ef = (meta.get(fn, {}) or {}).get('external_features', {}) or {}
        for k in ef.keys():
            key_set.add(k)
    keys = sorted(list(key_set))
    if not keys:
        logging.warning("No external features found in metadata.")
        return np.zeros((len(samples), 0), dtype=np.float32)
    rows = []
    for s in samples:
        fn = Path(s[1]).name
        ef = (meta.get(fn, {}) or {}).get('external_features', {}) or {}
        vec = [float(ef.get(k, 0.0)) if ef.get(k, None) is not None else 0.0 for k in keys]
        rows.append(np.array(vec, dtype=np.float32))
    return np.vstack(rows).astype(np.float32)

def ts_summary(window: pd.DataFrame) -> np.ndarray:
    w = window.copy()
    if len(w) >= 50:
        w = w.iloc[-50:]
    else:
        pad = 50 - len(w)
        w = pd.concat([w.iloc[[0]].repeat(pad), w]).iloc[-50:]
    close = w['close'].astype(float)
    ret = close.pct_change().fillna(0.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).fillna(0.0)
    # RSI
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi = ((rsi.fillna(50.0) - 50.0) / 50.0).fillna(0.0)
    # Min-max position
    roll_min = close.rolling(20).min().bfill()
    roll_max = close.rolling(20).max().bfill()
    pos = (close.iloc[-1] - roll_min.iloc[-1]) / (roll_max.iloc[-1] - roll_min.iloc[-1] + 1e-8)
    feats = np.array([
        ret.mean(), ret.std(), np.sum(np.abs(ret)), ret.iloc[-1],
        macd.iloc[-1], rsi.iloc[-1], float(pos)
    ], dtype=np.float32)
    return np.clip(feats, -5.0, 5.0)


def build_ts_features(samples: List[Tuple[datetime, str, int, str, float, str, int]]) -> np.ndarray:
    feats: List[np.ndarray] = []
    for s in samples:
        wfile = s[3]
        if not wfile:
            logging.warning(f"No window file for sample: {s}")
            feats.append(np.zeros(7, dtype=np.float32))
            continue
        wp = DATASET_DIR / 'windows' / wfile
        if not wp.exists():
            logging.warning(f"Window file not found: {wp}")
            feats.append(np.zeros(7, dtype=np.float32))
            continue
        try:
            w = pd.read_pickle(wp)
            # === EXPANDED FEATURES CACHE (no Part-0 redo) ===
            # Note: Part 1 does not re-load OHLCV per-asset globally. If cache exists per-asset,
            # it can be used downstream when building richer window stats. Kept no-op if absent.
            try:
                asset = s[5]
                os.makedirs("cache", exist_ok=True)
                _exp_path = os.path.join("cache", f"expanded_features_{asset}.parquet")
                if os.path.exists(_exp_path):
                    logging.info(f"Loading cached expanded features from {_exp_path}")
                    _ = pd.read_parquet(_exp_path)  # placeholder to trigger cache warmup
                else:
                    logging.info(f"Cache not found for asset {asset}, building features.")
            except Exception:
                logging.warning(f"Could not load or build expanded features for asset {s[5]}")
            feats.append(ts_summary(w))
        except Exception as e:
            logging.error(f"Error reading or processing window file {wp} for sample {s}: {e}")
            feats.append(np.zeros(7, dtype=np.float32))
    return np.vstack(feats).astype(np.float32)


def standardize_train_val(X_tr: np.ndarray, X_va: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_trs = (X_tr - mu) / sd
    X_vas = (X_va - mu) / sd
    return X_trs, X_vas, {'mu': mu, 'sd': sd}


def evaluate_calibrated_logreg(X_tr, y_tr, X_va, y_va) -> Tuple[object, Dict[str, float], np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, accuracy_score
    try:
        model = CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=2000), cv=3, method='sigmoid')
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        f1 = f1_score(y_va, pred, average='macro')
        acc = accuracy_score(y_va, pred)
        logging.info(f"Hybrid Calibrated LR | Acc={acc:.3f} F1={f1:.3f}")
        return model, {'acc': acc, 'f1': f1}, pred
    except Exception as e:
        logging.error(f"Error fitting or predicting with calibrated LR: {e}")
        return None, {}, np.zeros(len(y_va))


def backtest_gated(model, X_va: np.ndarray, y_va: np.ndarray, changes: np.ndarray) -> None:
    # y: 0=bear,1=bull,2=neutral; we trade long on 1, short on 0, hold on 2
    if not hasattr(model, 'predict_proba'):
        logging.warning('Backtest skipped: model lacks predict_proba')
        return
    try:
        probs = model.predict_proba(X_va)
        best = None
        for conf_thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            for diff_thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                pnl = []
                for p, chg in zip(probs, changes):
                    pbear, pbull, pneu = p[0], p[1], p[2]
                    top = max(pbear, pbull, pneu)
                    # require top confidence and directional separation if not neutral
                    if top < conf_thr or (top != pneu and abs(pbull - pbear) < diff_thr):
                        pnl.append(0.0); continue
                    if top == pbull and pbull > pbear:
                        pnl.append(float(chg) - 0.001)
                    elif top == pbear and pbear > pbull:
                        pnl.append(float(-chg) - 0.001)
                    else:
                        pnl.append(0.0)
                pnl = np.array(pnl, dtype=np.float32)
                if len(pnl) == 0:
                    continue
                eq = np.cumsum(pnl)
                roll_max = np.maximum.accumulate(eq)
                dd = roll_max - eq
                max_dd = float(dd.max()) if len(dd) else 0.0
                sharpe = float(pnl.mean() / (pnl.std() + 1e-8))
                win_rate = float((pnl > 0).mean())
                total = float(eq[-1]) if len(eq) else 0.0
                stat = {'conf': conf_thr, 'diff': diff_thr, 'pnl': total, 'sharpe': sharpe, 'win_rate': win_rate, 'max_dd': max_dd}
                if best is None or stat['pnl'] > best['pnl'] or (abs(stat['pnl'] - best['pnl']) < 1e-6 and sharpe > best['sharpe']):
                    best = stat
        if best:
            logging.info(f"Backtest (val): PnL={best['pnl']:.4f} Sharpe={best['sharpe']:.2f} WinRate={best['win_rate']:.2%} MaxDD={best['max_dd']:.4f} at conf={best['conf']} diff={best['diff']}")
        else:
            logging.warning('Backtest produced no trades with the given thresholds.')
    except Exception as e:
        logging.error(f"Error during backtest_gated: {e}")


def build_context_features(samples: List[Tuple[datetime, str, int, str, float, str, int]]) -> np.ndarray:
    assets = sorted(list({s[5] for s in samples}))
    asset_index = {a: i for i, a in enumerate(assets)}
    intervals = sorted(list({s[6] for s in samples}))
    interval_index = {iv: i for i, iv in enumerate(intervals)}
    ctx = []
    for s in samples:
        a_oh = np.zeros(len(assets), dtype=np.float32)
        iv_oh = np.zeros(len(intervals), dtype=np.float32)
        a_oh[asset_index.get(s[5], 0)] = 1.0
        iv_oh[interval_index.get(s[6], intervals[0] if intervals else 0)] = 1.0
        ctx.append(np.concatenate([a_oh, iv_oh], axis=0))
    return np.vstack(ctx).astype(np.float32)


def evaluate_binary_pipeline(X_tr, y_tr, X_va, y_va, changes) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, accuracy_score
    # filter to bull/bear only
    tr_mask = (y_tr != 2)
    va_mask = (y_va != 2)
    if tr_mask.sum() < 100 or va_mask.sum() < 100:
        logging.warning('Binary pipeline skipped: insufficient bull/bear samples.')
        return
    Xtr, ytr = X_tr[tr_mask], y_tr[tr_mask]
    Xva, yva = X_va[va_mask], y_va[va_mask]
    cva = changes[va_mask]
    # remap to 0/1 with 0=bear,1=bull
    ytr_bin = (ytr == 1).astype(np.int64)
    yva_bin = (yva == 1).astype(np.int64)
    try:
        model = CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=2000), cv=3, method='sigmoid')
        model.fit(Xtr, ytr_bin)
        pred = model.predict(Xva)
        f1 = f1_score(yva_bin, pred, average='macro')
        acc = accuracy_score(yva_bin, pred)
        logging.info(f"Binary Calibrated LR | Acc={acc:.3f} F1={f1:.3f} (bull/bear only)")
        # gated backtest for binary
        probs = model.predict_proba(Xva)[:, 1]  # prob bull
        best = None
        for conf_thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
            for margin in [0.0, 0.05, 0.10, 0.15]:
                pnl = []
                for p, chg in zip(probs, cva):
                    if abs(p - 0.5) < margin or max(p, 1 - p) < conf_thr:
                        pnl.append(0.0); continue
                    if p > 0.5:
                        pnl.append(float(chg) - 0.001)
                    else:
                        pnl.append(float(-chg) - 0.001)
                pnl = np.array(pnl, dtype=np.float32)
                if len(pnl) == 0:
                    continue
                eq = np.cumsum(pnl)
                roll_max = np.maximum.accumulate(eq)
                dd = roll_max - eq
                max_dd = float(dd.max()) if len(dd) else 0.0
                sharpe = float(pnl.mean() / (pnl.std() + 1e-8))
                win_rate = float((pnl > 0).mean())
                total = float(eq[-1]) if len(eq) else 0.0
                stat = {'conf': conf_thr, 'margin': margin, 'pnl': total, 'sharpe': sharpe, 'win_rate': win_rate, 'max_dd': max_dd}
                if best is None or stat['pnl'] > best['pnl'] or (abs(stat['pnl'] - best['pnl']) < 1e-6 and sharpe > best['sharpe']):
                    best = stat
        if best:
            logging.info(f"Binary Backtest (val): PnL={best['pnl']:.4f} Sharpe={best['sharpe']:.2f} WinRate={best['win_rate']:.2%} MaxDD={best['max_dd']:.4f} at conf={best['conf']} margin={best['margin']}")
        else:
            logging.warning('Binary backtest produced no trades with the given thresholds.')
    except Exception as e:
        logging.error(f"Error during binary pipeline evaluation: {e}")


def main():
    meta = load_metadata()
    samples = list_samples(meta)
    logging.info(f"Total samples: {len(samples)}")
    if len(samples) < 500:
        logging.warning('Not enough samples for hybrid training.')
        return

    train_idx, val_idx = train_val_split(samples, VAL_FRACTION, EMBARGO_BARS)
    logging.info(f"Train/Val sizes: {len(train_idx)}/{len(val_idx)} (embargo={EMBARGO_BARS})")

    # Build features
    X_dino = build_dino_features(samples)
    X_yolo = build_yolo_pseudo_features(samples, meta)
    X_ts = build_ts_features(samples)
    X_ctx = build_context_features(samples)
    X_ext = build_external_features(samples, meta)
    Y = np.array([s[2] for s in samples], dtype=np.int64)
    changes = np.array([s[4] for s in samples], dtype=np.float32)

    X_full = np.concatenate([X_dino, X_ts, X_yolo, X_ctx, X_ext], axis=1).astype(np.float32)
    logging.info(f"Feature blocks -> DINO:{X_dino.shape[1]}, TS:{X_ts.shape[1]}, YOLO:{X_yolo.shape[1]}, CTX:{X_ctx.shape[1]} | X:{X_full.shape}")

    X_tr = X_full[train_idx]; y_tr = Y[train_idx]
    X_va = X_full[val_idx]; y_va = Y[val_idx]
    c_va = changes[val_idx]

    # Standardize
    X_trs, X_vas, scl = standardize_train_val(X_tr, X_va)

    # Train calibrated LR (multiclass)
    model, metrics, _ = evaluate_calibrated_logreg(X_trs, y_tr, X_vas, y_va)

    # Backtest on val
    backtest_gated(model, X_vas, y_va, c_va)

    # Binary variant (bull/bear only)
    evaluate_binary_pipeline(X_trs, y_tr, X_vas, y_va, c_va)

    # Save artifacts
    out_dir = BASE_DIR / 'hybrid_models'
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({'mu': scl['mu'], 'sd': scl['sd']}, out_dir / 'scaler.pkl')
        joblib.dump(model, out_dir / 'calibrated_logreg.pkl')
        logging.info(f"Saved scaler and model to {out_dir}")
    except Exception as e:
        logging.error(f"Error saving artifacts: {e}")

    # Minimal run summary
    try:
        logging.info("\n=== RUN SUMMARY ===")
        try:
            print(f"Sharpe (val): {metrics.get('sharpe_val'):.3f}")
        except Exception:
            pass
        try:
            print(f"StageA F1: {metrics.get('stageA_f1'):.3f} | StageB F1: {metrics.get('stageB_f1'):.3f}")
        except Exception:
            pass
        try:
            print(f"Trades: {metrics.get('trades_total')}")
        except Exception:
            pass
        logging.info("Done.")
    except Exception:
        pass


if __name__ == '__main__':
    main()


