"""
Part 1B: Two-stage hybrid and improved backtest (fast, no regeneration)

Stage A (Movement): neutral vs move (bear/bull)
Stage B (Direction): bear vs bull (on movement-only samples)

Features: DINO embeddings + TS summaries + YOLO pseudo + context (asset/interval)
Backtest: trade only if StageA_conf >= m_conf and |pbull - pbear| >= diff
"""




import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# === NEW IMPORTS (idempotent) ===
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore
try:
    from freedom44.utils.seed import set_all_seeds  # type: ignore
except Exception:
    try:
        from utils.seed import set_all_seeds  # type: ignore
    except Exception:
        def set_all_seeds(seed: int) -> None:
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
    from freedom44.cv.purged import purged_time_splits  # type: ignore
except Exception:
    try:
        from cv.purged import purged_time_splits  # type: ignore
    except Exception:
        purged_time_splits = None  # type: ignore
try:
    from freedom44.features.microstructure import add_microstructure  # type: ignore
    from freedom44.features.volatility import add_har_rv  # type: ignore
    from freedom44.features.regime import add_regime  # type: ignore
    from freedom44.features.funding_basis import add_funding_basis  # type: ignore
except Exception:
    try:
        from features.microstructure import add_microstructure  # type: ignore
        from features.volatility import add_har_rv  # type: ignore
        from features.regime import add_regime  # type: ignore
        from features.funding_basis import add_funding_basis  # type: ignore
    except Exception:
        def add_microstructure(df):
            return pd.DataFrame(index=getattr(df, 'index', None))
        def add_har_rv(rets):
            return pd.DataFrame(index=getattr(rets, 'index', None))
        def add_regime(close):
            return pd.DataFrame(index=getattr(close, 'index', None))
        def add_funding_basis(ohlcv, funding=None, basis=None):
            return pd.DataFrame(index=getattr(ohlcv, 'index', None))
try:
    from freedom44.models.gbm import make_lgbm_movement, make_lgbm_direction  # type: ignore
except Exception:
    try:
        from models.gbm import make_lgbm_movement, make_lgbm_direction  # type: ignore
    except Exception:
        def make_lgbm_movement():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)
        def make_lgbm_direction():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)
try:
    from freedom44.models.stacking import ProbabilityStacker  # type: ignore
except Exception:
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
            pass
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





BASE_DIR = Path(os.environ.get('HYBRID_BASE_DIR', '/content/hybrid_workspace'))
DATASET_DIR = BASE_DIR / 'datasets' / 'chart_dataset'
EMBED_DIR = DATASET_DIR / 'embeddings_dino'
WINDOWS_DIR = DATASET_DIR / 'windows'
WINDOWS_FWD_DIR = DATASET_DIR / 'windows_fwd'
HYBRID_MODELS_DIR = BASE_DIR / 'hybrid_models'
PER_ASSET_OVR_JSON = HYBRID_MODELS_DIR / 'per_asset_optuna.json'

VAL_FRACTION = 0.2
EMBARGO_BARS = 10
USE_MLP = True  # enable scikit-learn MLPClassifier for Stage A/B
# Embedding backend (DINOv2 default; allow DINOv3 or custom via HF model id)
EMBED_MODEL_ID = os.environ.get('EMBED_MODEL_ID', 'facebook/dinov2-base')
EMBED_DIR_OVERRIDE = os.environ.get('EMBED_DIR_OVERRIDE', '')
EMBED_DIR_V2 = DATASET_DIR / 'embeddings_dino'
EMBED_DIR_V3 = DATASET_DIR / 'embeddings_dinov3'
ACTIVE_EMBED_DIR = Path(EMBED_DIR_OVERRIDE) if EMBED_DIR_OVERRIDE else (EMBED_DIR_V3 if 'dinov3' in EMBED_MODEL_ID.lower() else EMBED_DIR_V2)


# Sentiment config (FinBERT)
USE_SENTIMENT = True
SENT_JSON = os.environ.get('SENT_JSON', str(BASE_DIR / 'datasets' / 'sentiment.json'))
SENT_WINDOW_MINUTES = 240  # aggregate headlines within this lookback before sample timestamp
HF_DATASET = os.environ.get('HF_SENT_DATASET', 'flowfree/crypto-news-headlines')

# Optional: overrides from Optuna tuning
OPTUNA_JSON = Path(os.environ.get('OPTUNA_JSON', str(BASE_DIR / 'hybrid_models' / 'optuna_best.json')))
OVR = {}
if OPTUNA_JSON.exists():
    try:
        import json as _json
        OVR = _json.loads(OPTUNA_JSON.read_text())
        print('Loaded Optuna overrides:', OVR)
    except Exception as _e:
        print('Optuna overrides load failed:', _e)

# Training/trading scopes (interval minutes)
TRAIN_ONLY_INTERVALS = [int(x) for x in os.environ.get('TRAIN_ONLY_INTERVALS', '240').split(',') if x.strip()]
TRADE_ONLY_INTERVALS = [int(x) for x in os.environ.get('TRADE_ONLY_INTERVALS', '240').split(',') if x.strip()]

# Percentile gating defaults (can be overridden by OVR)
PM_Q = float(OVR.get('pm_q', os.environ.get('PM_Q', 0.70)))
MARGIN_Q = float(OVR.get('margin_q', os.environ.get('MARGIN_Q', 0.70)))
ENSEMBLE_SEEDS = int(os.environ.get('ENSEMBLE_SEEDS', '3'))
SMOOTH_ALPHA = float(os.environ.get('SMOOTH_ALPHA', '0.0'))  # 0.0 disables smoothing
GRID_MCONF = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GRID_DIFF = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
MIN_TRADES = int(os.environ.get('MIN_TRADES', '50'))
MAX_DD = float(os.environ.get('MAX_DD', '0.5'))
WHIP_LOOKBACK = int(os.environ.get('WHIP_LOOKBACK', '2'))
WHIP_MARGIN_BOOST = float(os.environ.get('WHIP_MARGIN_BOOST', '0.02'))
SIZE_FLOOR = float(os.environ.get('SIZE_FLOOR', '0.02'))
SIZE_CAP = float(os.environ.get('SIZE_CAP', '0.25'))
FEE = float(os.environ.get('FEE', '0.001'))
VOL_INV_K = float(os.environ.get('VOL_INV_K', '0.02'))  # sizing scales down in high vol
MIN_ASSET_TRAIN = int(os.environ.get('MIN_ASSET_TRAIN', '300'))
SHORT_MARGIN_EXTRA = float(os.environ.get('SHORT_MARGIN_EXTRA', '0.02'))
LONG_SIZE_MULT = float(os.environ.get('LONG_SIZE_MULT', '1.10'))
SHORT_SIZE_MULT = float(os.environ.get('SHORT_SIZE_MULT', '0.80'))
COOLDOWN_BARS = int(os.environ.get('COOLDOWN_BARS', '1'))
TRADES_PER_MONTH_CAP = int(os.environ.get('TRADES_PER_MONTH_CAP', '20'))
TP_MULT = float(os.environ.get('TP_MULT', '2.0'))  # take-profit multiple of ts_std
SL_MULT = float(os.environ.get('SL_MULT', '1.0'))  # stop-loss multiple of ts_std
MIN_TRADES_PER_ASSET = int(os.environ.get('MIN_TRADES_PER_ASSET', '10'))
MCONF_MIN = float(os.environ.get('MCONF_MIN', '0.35'))
MCONF_MAX = float(os.environ.get('MCONF_MAX', '0.90'))
DIFF_MIN = float(os.environ.get('DIFF_MIN', '0.01'))
DIFF_MAX = float(os.environ.get('DIFF_MAX', '0.35'))
TREND_Q = float(os.environ.get('TREND_Q', '0.70'))
TREND_DIFF_REDUCTION = float(os.environ.get('TREND_DIFF_REDUCTION', '0.02'))
CHOP_DIFF_ADD = float(os.environ.get('CHOP_DIFF_ADD', '0.01'))
REQUIRE_TREND_ALIGN = int(os.environ.get('REQUIRE_TREND_ALIGN', '0'))
SELECT_SHARPE_MIN = float(os.environ.get('SELECT_SHARPE_MIN', '0.20'))
SELECT_PNL_MIN = float(os.environ.get('SELECT_PNL_MIN', '0.0'))
SELECT_TOP_N = int(os.environ.get('SELECT_TOP_N', '0'))
SELECT_TOP_BY = os.environ.get('SELECT_TOP_BY', 'composite')  # 'sharpe' | 'pnl' | 'composite'
SELECT_ALPHA = float(os.environ.get('SELECT_ALPHA', '1.0'))   # weight for pnl in composite score
SELECT_BETA_TRADES = float(os.environ.get('SELECT_BETA_TRADES', '0.0'))   # weight for trades in composite
TP_TREND_BOOST = float(os.environ.get('TP_TREND_BOOST', '0.25'))
TP_CHOP_CUT = float(os.environ.get('TP_CHOP_CUT', '0.15'))
SL_TREND_REL = float(os.environ.get('SL_TREND_REL', '1.00'))
SL_CHOP_REL = float(os.environ.get('SL_CHOP_REL', '0.90'))
STAB_MCONF_JITTER = float(os.environ.get('STAB_MCONF_JITTER', '0.05'))
STAB_DIFF_JITTER = float(os.environ.get('STAB_DIFF_JITTER', '0.01'))
STAB_SHARPE_MIN = float(os.environ.get('STAB_SHARPE_MIN', '0.00'))
STAB_WF_MIN_SHARPE = float(os.environ.get('STAB_WF_MIN_SHARPE', '0.15'))
ADX_MIN_TREND = float(os.environ.get('ADX_MIN_TREND', '18.0'))
BB_SQUEEZE_Q = float(os.environ.get('BB_SQUEEZE_Q', '0.30'))
CHOP_DIFF_ADD_EXTRA = float(os.environ.get('CHOP_DIFF_ADD_EXTRA', '0.02'))
SKIP_LOW_ADX_TREND = int(os.environ.get('SKIP_LOW_ADX_TREND', '1'))
USE_OVR_GATES = int(os.environ.get('USE_OVR_GATES', '0'))  # include Optuna m_conf/diff in grid
WF_FOLDS = int(os.environ.get('WF_FOLDS', '5'))             # walk-forward folds on val tail
USE_WF_GRID_SELECT = int(os.environ.get('USE_WF_GRID_SELECT', '1'))  # choose gates by WF mean/median Sharpe
WF_USE_MEDIAN = int(os.environ.get('WF_USE_MEDIAN', '0'))            # 1 -> use median Sharpe instead of mean
MIN_TRADES_PER_FOLD = int(os.environ.get('MIN_TRADES_PER_FOLD', '5'))
FEE_SENSITIVITY = os.environ.get('FEE_SENSITIVITY', '')      # e.g., '0.001,0.0025'
SEED_OFFSET = int(os.environ.get('SEED_OFFSET', '0'))        # global seed offset for multi-seed runs
USE_TIME_STOP = int(os.environ.get('USE_TIME_STOP', '1'))
TIME_STOP_BARS = int(os.environ.get('TIME_STOP_BARS', '10'))
USE_TRAIL_STOP = int(os.environ.get('USE_TRAIL_STOP', '1'))
TRAIL_MULT = float(os.environ.get('TRAIL_MULT', '1.0'))      # trailing threshold in sigma units
SPREAD = float(os.environ.get('SPREAD', '0.0000'))            # extra per-trade cost (not scaled by size)
SELECT_MIN_TRADES = int(os.environ.get('SELECT_MIN_TRADES', '20'))
SELECT_MAX_DD = float(os.environ.get('SELECT_MAX_DD', '0.75'))


def load_metadata() -> Dict[str, dict]:
    with open(DATASET_DIR / 'metadata.json', 'r') as f:
        return json.load(f)


def list_samples(meta: Dict[str, dict]) -> List[Tuple[datetime, str, int, str, float, str, int]]:
    label_map = {'bearish': 0, 'bullish': 1, 'neutral': 2}
    samples = []
    for fname, m in meta.items():
        lab = m.get('label');  
        if lab not in label_map: 
            continue
        ts = datetime.fromisoformat(m['timestamp'])
        img = DATASET_DIR / lab / fname
        if not img.exists():
            continue
        samples.append((ts, str(img), label_map[lab], m.get('window_file'), float(m.get('avg_change', 0.0)), m.get('asset','UNK'), int(m.get('interval_min',0))))
    samples.sort(key=lambda x: x[0])
    return samples


def split_idx(n: int, val_fraction: float, embargo: int) -> Tuple[List[int], List[int]]:
    split = int((1.0 - val_fraction) * n)
    return list(range(0, max(0, split - embargo))), list(range(min(n, split + embargo), n))


_EMBED_PROCESSOR = None
_EMBED_MODEL = None
_EMBED_DEVICE = None
_EMBED_DIM: Optional[int] = None


def _candidate_embed_ids(primary_id: str) -> List[str]:
    """Return a list of candidate HF repo IDs to try for DINO models."""
    pid = (primary_id or '').strip()
    # If user passed a shorthand, map to a likely official repo
    alias_map = {
        'dinov3-base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'dinov3-small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'dinov3-large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'facebook/dinov3-base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    }
    if pid in alias_map:
        pid = alias_map[pid]
    # Ensure facebook/ prefix if looks like dinov3-* without org
    if pid and 'dinov3' in pid and '/' not in pid:
        pid = f'facebook/{pid}'
    # Try provided id first, then reasonable defaults
    cands = []
    if pid:
        cands.append(pid)
    cands += [
        'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'facebook/dinov3-vits16-pretrain-lvd1689m',
        'facebook/dinov2-base',
    ]
    # Deduplicate while preserving order
    seen = set(); result = []
    for x in cands:
        if x and x not in seen:
            seen.add(x); result.append(x)
    return result


def _load_embedder(model_id: str):
    global _EMBED_PROCESSOR, _EMBED_MODEL, _EMBED_DEVICE, _EMBED_DIM
    if _EMBED_MODEL is not None:
        return
    from transformers import AutoImageProcessor, AutoModel
    import torch
    token = os.environ.get('HF_TOKEN', None)
    kwargs = {'token': token} if token else {}
    last_err = None
    for mid in _candidate_embed_ids(model_id):
        try:
            _EMBED_PROCESSOR = AutoImageProcessor.from_pretrained(mid, **kwargs)
            _EMBED_MODEL = AutoModel.from_pretrained(mid, **kwargs)
            _EMBED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _EMBED_MODEL = _EMBED_MODEL.to(_EMBED_DEVICE).eval()
            _EMBED_DIM = None
            print(f"Loaded embedder: {mid}")
            return
        except Exception as e:
            last_err = e
            continue
    # Final fallback to dinov2-base without token
    try:
        _EMBED_PROCESSOR = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        _EMBED_MODEL = AutoModel.from_pretrained('facebook/dinov2-base')
        _EMBED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _EMBED_MODEL = _EMBED_MODEL.to(_EMBED_DEVICE).eval()
        _EMBED_DIM = None
        print("Loaded fallback embedder: facebook/dinov2-base")
    except Exception as e:
        raise RuntimeError(f"Failed to load any embedding model. Last error: {last_err or e}")


def _compute_image_embedding(img_path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        import torch
        _load_embedder(EMBED_MODEL_ID)
        img = Image.open(str(img_path)).convert('RGB').resize((448, 448))
        inputs = _EMBED_PROCESSOR(images=img, return_tensors='pt')
        inputs = {k: v.to(_EMBED_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = _EMBED_MODEL(**inputs)
        # Try standard last_hidden_state; fallback to pooled if available
        x = getattr(out, 'last_hidden_state', None)
        if x is None:
            x = getattr(out, 'pooler_output', None)
        if x is None:
            return None
        feats = x.mean(dim=1) if x.ndim == 3 else x
        feats = torch.nn.functional.normalize(feats, p=2, dim=1).squeeze(0).float().cpu().numpy()
        global _EMBED_DIM
        if _EMBED_DIM is None:
            _EMBED_DIM = int(feats.shape[0])
        return feats.astype(np.float32)
    except Exception as e:
        print(f"Embedding compute failed for {img_path.name}: {e}")
        return None


def load_dino_emb(path: Path) -> np.ndarray:
    import torch
    t = torch.load(str(path))
    return (t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)).astype(np.float32)


def build_dino_features(samples) -> np.ndarray:
    feats = []
    active_dir = ACTIVE_EMBED_DIR
    active_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        img_path = Path(s[1])
        stem = img_path.stem
        pt = active_dir / f"{stem}.pt"
        if pt.exists():
            try:
                feats.append(load_dino_emb(pt))
                continue
            except Exception:
                pass
        # If no embedding found, compute on-the-fly and cache
        vec = _compute_image_embedding(img_path)
        if vec is not None:
            try:
                import torch
                torch.save(torch.from_numpy(vec), str(pt))
            except Exception:
                pass
            feats.append(vec)
        else:
            dim = _EMBED_DIM if _EMBED_DIM is not None else 768
            feats.append(np.zeros(dim, dtype=np.float32))
    # Ensure rectangular by padding/truncating as needed
    max_dim = max(v.shape[0] for v in feats)
    feats_p = [v if v.shape[0] == max_dim else np.pad(v, (0, max_dim - v.shape[0])) for v in feats]
    return np.vstack(feats_p).astype(np.float32)


def build_yolo_pseudo_features(samples, meta) -> np.ndarray:
    names = ['fvg_bull','fvg_bear','ob_bull','ob_bear','bos_bull','bos_bear','choch_bull','choch_bear']
    feats = []
    for s in samples:
        fn = Path(s[1]).name
        boxes = meta.get(fn, {}).get('yolo_pseudo_boxes', [])
        v = np.zeros(2*len(names), dtype=np.float32)
        for b in boxes:
            try:
                c, _, _, w, h = b
                if 0 <= c < len(names):
                    v[c] += 1.0
                    v[len(names)+c] += float(w) * float(h)
            except Exception:
                continue
        feats.append(v)
    return np.vstack(feats).astype(np.float32)


def ts_summary(window: pd.DataFrame) -> np.ndarray:
    w = window.copy()
    if len(w) >= 50:
        w = w.iloc[-50:]
    else:
        w = pd.concat([w.iloc[[0]].repeat(50-len(w)), w]).iloc[-50:]
    close = w['close'].astype(float)
    high = w['high'].astype(float)
    low = w['low'].astype(float)
    ret = close.pct_change().fillna(0.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).fillna(0.0)
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi = ((rsi.fillna(50.0) - 50.0) / 50.0).fillna(0.0)
    roll_min = close.rolling(20).min().bfill(); roll_max = close.rolling(20).max().bfill()
    pos = (close.iloc[-1] - roll_min.iloc[-1]) / (roll_max.iloc[-1] - roll_min.iloc[-1] + 1e-8)
    # ADX (approximate Wilder's method, 14)
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)
    plus_dm = (high.diff().clip(lower=0.0)).fillna(0.0)
    minus_dm = (-low.diff().clip(upper=0.0)).fillna(0.0)
    n = 14
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = (100.0 * (plus_dm.ewm(alpha=1.0/n, adjust=False).mean() / (atr + 1e-8))).fillna(0.0)
    minus_di = (100.0 * (minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / (atr + 1e-8))).fillna(0.0)
    dx = (100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)).fillna(0.0)
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean().fillna(0.0)
    adx_last = float(adx.iloc[-1])
    # Bollinger bandwidth (20)
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    bb_bw = float(((ma20 + 2*sd20) - (ma20 - 2*sd20)).iloc[-1] / (ma20.iloc[-1] + 1e-8)) if ma20.notna().iloc[-1] and sd20.notna().iloc[-1] else 0.0
    vec = np.array([
        ret.mean(), ret.std(), np.sum(np.abs(ret)), ret.iloc[-1],
        macd.iloc[-1], rsi.iloc[-1], float(pos), adx_last, bb_bw
    ], dtype=np.float32)
    return np.clip(vec, -5.0, 5.0)


def build_ts_features(samples) -> np.ndarray:
    feats = []
    for s in samples:
        wfile = s[3]
        if not wfile:
            feats.append(np.zeros(9, dtype=np.float32)); continue
        wp = DATASET_DIR / 'windows' / wfile
        if not wp.exists():
            feats.append(np.zeros(9, dtype=np.float32)); continue
        try:
            feats.append(ts_summary(pd.read_pickle(wp)))
        except Exception:
            feats.append(np.zeros(9, dtype=np.float32))
    return np.vstack(feats).astype(np.float32)


def build_context_features(samples) -> np.ndarray:
    assets = sorted({s[5] for s in samples}); asset_idx = {a:i for i,a in enumerate(assets)}
    ivals = sorted({s[6] for s in samples}); iv_idx = {iv:i for i,iv in enumerate(ivals)}
    V = []
    for s in samples:
        a = np.zeros(len(assets), dtype=np.float32); a[asset_idx[s[5]]] = 1.0
        iv = np.zeros(len(ivals), dtype=np.float32); iv[iv_idx[s[6]]] = 1.0
        V.append(np.concatenate([a, iv]))
    return np.vstack(V).astype(np.float32)


def build_external_features(samples, meta: Dict[str, dict]) -> np.ndarray:
    """
    Build a numeric feature matrix from metadata['external_features'] per image.
    Uses a fixed sorted key order discovered over all samples to keep columns stable.
    Missing values -> 0.0.
    """
    # Discover keys
    keys: List[str] = []
    key_set = set()
    for s in samples:
        fn = Path(s[1]).name
        m = meta.get(fn, {})
        ef = m.get('external_features', {}) or {}
        for k in ef.keys():
            if k not in key_set:
                key_set.add(k)
    keys = sorted(list(key_set))
    if not keys:
        return np.zeros((len(samples), 0), dtype=np.float32)
    # Build rows
    rows: List[np.ndarray] = []
    for s in samples:
        fn = Path(s[1]).name
        m = meta.get(fn, {})
        ef = m.get('external_features', {}) or {}
        vec = [float(ef.get(k, 0.0)) if ef.get(k, None) is not None else 0.0 for k in keys]
        rows.append(np.array(vec, dtype=np.float32))
    X = np.vstack(rows).astype(np.float32)
    return X


def get_external_feature_keys(samples, meta: Dict[str, dict]) -> List[str]:
    key_set = set()
    for s in samples:
        fn = Path(s[1]).name
        ef = (meta.get(fn, {}) or {}).get('external_features', {}) or {}
        for k in ef.keys():
            key_set.add(k)
    return sorted(list(key_set))

def standardize(X_tr, X_va):
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0) + 1e-8
    return (X_tr - mu)/sd, (X_va - mu)/sd


def build_sentiment_features(samples, window_minutes: int = None, embargo_minutes: int = 0, use_future: bool = False) -> np.ndarray:
    if not USE_SENTIMENT:
        return np.zeros((len(samples), 5), dtype=np.float32)
    p = Path(SENT_JSON)
    if not p.exists():
        # Try to auto-build from Hugging Face dataset
        try:
            maybe_build_sentiment_from_hf()
        except Exception as e:
            print(f"Auto-build sentiment failed: {e}")
    if not p.exists():
        print(f"Sentiment file not found at {p}; using zeros.")
        return np.zeros((len(samples), 5), dtype=np.float32)
    try:
        raw = json.loads(Path(p).read_text())
    except Exception as e:
        print(f"Failed to load sentiment JSON: {e}; using zeros.")
        return np.zeros((len(samples), 5), dtype=np.float32)
    # Expect structure: {"YYYY-mm-ddTHH:MM:SS": {"pos":...,"neg":...,"neu":...,"sent":...}, ...}
    # Build a sorted list for fast window queries
    rows = []
    for k, v in raw.items():
        try:
            ts = datetime.fromisoformat(k)
            rows.append((ts, float(v.get('pos',0.0)), float(v.get('neg',0.0)), float(v.get('neu',0.0)), float(v.get('sent',0.0))))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0])
    tseries = [r[0] for r in rows]
    vals = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float32)

    def window_agg(t_end: datetime, lookback_min=SENT_WINDOW_MINUTES, embargo_min: int = 0, future: bool = False):
        if len(tseries) == 0:
            return np.zeros(5, dtype=np.float32)
        lb = lookback_min
        if future:
            t_start = t_end + pd.Timedelta(minutes=1)
            t_end_adj = t_start + pd.Timedelta(minutes=lb)
        else:
            t_end_adj = t_end - pd.Timedelta(minutes=embargo_min)
            t_start = t_end_adj - pd.Timedelta(minutes=lb)
        sel = [i for i, tt in enumerate(tseries) if t_start <= tt <= t_end_adj]
        if len(sel) == 0:
            return np.zeros(5, dtype=np.float32)
        v = vals[sel]
        mean = v.mean(axis=0)
        std = v.std(axis=0)
        # features: [pos_mean, neg_mean, neu_mean, sent_mean, sent_std]
        return np.array([mean[0], mean[1], mean[2], mean[3], std[3]], dtype=np.float32)

    feats = []
    wm = SENT_WINDOW_MINUTES if window_minutes is None else window_minutes
    for s in samples:
        feats.append(window_agg(s[0], lookback_min=wm, embargo_min=embargo_minutes, future=use_future))
    return np.vstack(feats).astype(np.float32)


def maybe_build_sentiment_from_hf():
    os.makedirs(BASE_DIR / 'datasets', exist_ok=True)
    out_path = Path(SENT_JSON)
    print(f"Building sentiment JSON from HF dataset '{HF_DATASET}' -> {out_path}")
    # Lazy imports and installs
    try:
        from datasets import load_dataset
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datasets'])
        from datasets import load_dataset
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', 'torch'])
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

    # Load a simple crypto news dataset
    ds = load_dataset(HF_DATASET)
    # Try common split names
    if 'train' in ds:
        d = ds['train']
    else:
        # take first available split
        first_key = list(ds.keys())[0]
        d = ds[first_key]

    import pandas as pd
    df = pd.DataFrame(d)
    # Heuristic: find text and timestamp columns
    txt_col = None
    for c in ['title', 'headline', 'text', 'content']:
        if c in df.columns:
            txt_col = c; break
    if txt_col is None:
        raise RuntimeError('No text column found in HF dataset')
    ts_col = None
    for c in ['timestamp', 'date', 'published', 'published_at', 'time']:
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        # fabricate timestamps spaced by minutes to allow aggregation
        base = pd.Timestamp('2022-01-01')
        df['timestamp'] = [base + pd.Timedelta(minutes=i) for i in range(len(df))]
        ts_col = 'timestamp'
    # Parse timestamps to tz-naive ISO strings
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[ts_col, txt_col])
    df[ts_col] = df[ts_col].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Run FinBERT
    tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    mdl = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdl = mdl.to(device)

    def softmax(x):
        e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        return e / e.sum(dim=-1, keepdim=True)

    texts = df[txt_col].astype(str).tolist()
    batch = 64
    pos_list, neg_list, neu_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            enc = tok(chunk, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            logits = mdl(**enc).logits  # [neu, neg, pos]
            probs = softmax(logits).cpu().numpy()
            neu_list += probs[:,0].tolist()
            neg_list += probs[:,1].tolist()
            pos_list += probs[:,2].tolist()

    df['pos'] = pos_list; df['neg'] = neg_list; df['neu'] = neu_list
    df['sent'] = (df['pos'] - df['neg']).clip(-1, 1)
    agg = df.groupby(ts_col, as_index=True)[['pos','neg','neu','sent']].mean()
    out = {ts: {'pos': float(r.pos), 'neg': float(r.neg), 'neu': float(r.neu), 'sent': float(r.sent)} for ts, r in agg.iterrows()}
    out_path.write_text(json.dumps(out))
    print(f"Built sentiment JSON with {len(out)} records.")


def main():
    meta = load_metadata()
    samples = list_samples(meta)
    print("Total samples:", len(samples))
    n = len(samples)
    tr_idx, va_idx = split_idx(n, VAL_FRACTION, EMBARGO_BARS)
    print(f"Train/Val sizes: {len(tr_idx)}/{len(va_idx)} (embargo={EMBARGO_BARS})")


    Xd = build_dino_features(samples)
    Xt = build_ts_features(samples)
    Xy = build_yolo_pseudo_features(samples, meta)
    Xc = build_context_features(samples)
    Xe = build_external_features(samples, meta)
    # Build three sentiment variants for trust checks
    Xs = build_sentiment_features(samples)
    Xs_emb = build_sentiment_features(samples, embargo_minutes=60)  # exclude last 60m to de-risk leakage
    Xs_future = build_sentiment_features(samples, use_future=True)  # intentionally wrong (future window)
    # Use embargoed sentiment in main X to reduce leakage risk
    X = np.concatenate([Xd, Xt, Xy, Xc, Xe, Xs_emb], axis=1).astype(np.float32)
    Y = np.array([s[2] for s in samples], dtype=np.int64)
    changes = np.array([s[4] for s in samples], dtype=np.float32)
    print("X shape:", X.shape)

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = Y[tr_idx], Y[va_idx]
    c_va = changes[va_idx]
    intervals_all = np.array([s[6] for s in samples], dtype=np.int64)
    times_all = np.array([s[0] for s in samples])
    iv_tr = intervals_all[tr_idx]
    iv_va = intervals_all[va_idx]
    # Restrict to configured intervals
    tr_keep = np.isin(iv_tr, TRAIN_ONLY_INTERVALS)
    va_keep = np.isin(iv_va, TRADE_ONLY_INTERVALS)
    X_tr, y_tr = X_tr[tr_keep], y_tr[tr_keep]
    X_va, y_va = X_va[va_keep], y_va[va_keep]
    c_va = c_va[va_keep]
    iv_va = iv_va[va_keep]
    t_va = times_all[va_idx][va_keep]
    # TS volatility proxy (std of returns) for EV gating
    ts_std_all = Xt[:, 1]
    ts_std_va = ts_std_all[va_idx][va_keep]
    X_trs, X_vas = standardize(X_tr, X_va)
    # Save scaler for deployment (mu/sd on training subset)
    try:
        import joblib
        mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0) + 1e-8
        (BASE_DIR / 'hybrid_models').mkdir(parents=True, exist_ok=True)
        joblib.dump({'mu': mu, 'sd': sd}, BASE_DIR / 'hybrid_models' / 'scaler_part1b.pkl')
    except Exception as _e:
        print('Scaler save failed:', _e)

    # Build trust-check sentiment variants aligned to filtered splits
    X_tr_emb = np.concatenate([
        Xd[tr_idx][tr_keep], Xt[tr_idx][tr_keep], Xy[tr_idx][tr_keep], Xc[tr_idx][tr_keep], Xe[tr_idx][tr_keep], Xs_emb[tr_idx][tr_keep]
    ], axis=1)
    X_va_emb = np.concatenate([
        Xd[va_idx][va_keep], Xt[va_idx][va_keep], Xy[va_idx][va_keep], Xc[va_idx][va_keep], Xe[va_idx][va_keep], Xs_emb[va_idx][va_keep]
    ], axis=1)
    X_tr_fut = np.concatenate([
        Xd[tr_idx][tr_keep], Xt[tr_idx][tr_keep], Xy[tr_idx][tr_keep], Xc[tr_idx][tr_keep], Xe[tr_idx][tr_keep], Xs_future[tr_idx][tr_keep]
    ], axis=1)
    X_va_fut = np.concatenate([
        Xd[va_idx][va_keep], Xt[va_idx][va_keep], Xy[va_idx][va_keep], Xc[va_idx][va_keep], Xe[va_idx][va_keep], Xs_future[va_idx][va_keep]
    ], axis=1)
    X_trs_emb, X_vas_emb = standardize(X_tr_emb, X_va_emb)
    X_trs_fut, X_vas_fut = standardize(X_tr_fut, X_va_fut)

    # Regime features for gating (use raw TS summary on val)
    ts_va_feats = Xt[va_idx][va_keep]
    # trend strength proxy: ADX (new) and |MACD| + |RSI|
    adx_col = 7 if ts_va_feats.shape[1] >= 8 else None
    bb_col = 8 if ts_va_feats.shape[1] >= 9 else None
    trend_strength = (np.abs(ts_va_feats[:, 4]) + np.abs(ts_va_feats[:, 5]))
    if adx_col is not None:
        trend_strength = trend_strength + (ts_va_feats[:, adx_col] / 100.0)
    trend_thr = np.quantile(trend_strength, TREND_Q) if len(trend_strength) else 0.0
    bb_squeeze_thr = None
    if bb_col is not None:
        bb_vals = ts_va_feats[:, bb_col]
        bb_squeeze_thr = float(np.quantile(bb_vals, BB_SQUEEZE_Q))

    # Load forward paths (optional, for time-stop/trailing). Build only for validation subset
    # Also keep a window path map for ATR/entry context
    future_map = {fn: m.get('future_file') for fn, m in meta.items()}
    window_map = {fn: m.get('window_file') for fn, m in meta.items()}
    # Build list aligned to validation indices after filtering
    fwd_returns: List[Optional[np.ndarray]] = [None] * len(y_va)
    fwd_entry_price: List[Optional[float]] = [None] * len(y_va)
    window_paths: List[Optional[Path]] = [None] * len(y_va)
    try:
        va_positions = np.where(va_keep)[0]
        for idx_local_in_va, pos_in_va in enumerate(va_positions):
            sample_idx = va_idx[pos_in_va]
            s = samples[sample_idx]
            img_fn = Path(s[1]).name
            future_fn = future_map.get(img_fn)
            if not future_fn:
                continue
            fwd_path = WINDOWS_FWD_DIR / future_fn
            if not fwd_path.exists():
                continue
            df_fwd = pd.read_pickle(fwd_path)
            if 'close' not in df_fwd.columns or len(df_fwd) == 0:
                continue
            base = float(df_fwd['close'].iloc[0])
            if base <= 0:
                continue
            ret = (df_fwd['close'].astype(float) / base) - 1.0
            fwd_returns[pos_in_va] = ret.values.astype(np.float32)
            fwd_entry_price[pos_in_va] = base
            # window path
            w_fn = window_map.get(img_fn)
            if w_fn:
                wp = DATASET_DIR / 'windows' / w_fn
                if wp.exists():
                    window_paths[pos_in_va] = wp
    except Exception as _e:
        fwd_returns = [None] * len(y_va)
        fwd_entry_price = [None] * len(y_va)
        window_paths = [None] * len(y_va)

    # Stage A: neutral(0) vs move(1)
    y_tr_m = (y_tr != 2).astype(np.int64)
    y_va_m = (y_va != 2).astype(np.int64)
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neural_network import MLPClassifier
    def balance_binary(Xb, yb):
        idx0 = np.where(yb == 0)[0]; idx1 = np.where(yb == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            return Xb, yb
        if len(idx0) > len(idx1):
            add = np.random.choice(idx1, len(idx0) - len(idx1), replace=True)
            idx = np.concatenate([np.arange(len(yb)), add])
        else:
            add = np.random.choice(idx0, len(idx1) - len(idx0), replace=True)
            idx = np.concatenate([np.arange(len(yb)), add])
        return Xb[idx], yb[idx]

    if USE_MLP:
        hidden = int(OVR.get('hidden', 256))
        lr = float(OVR.get('lr', 1e-3))
        def make_mlp(seed: int):
            return MLPClassifier(
                hidden_layer_sizes=(hidden,), activation='relu', alpha=1e-4,
                batch_size=128, learning_rate_init=lr, max_iter=80, early_stopping=True,
                random_state=seed
            )
        from sklearn.calibration import CalibratedClassifierCV
        # Stage A ensemble
        X_mov, y_mov = balance_binary(X_trs, y_tr_m)
        pm_list = []
        mov_cals = []
        for seed in range(ENSEMBLE_SEEDS):
            np.random.seed(42 + seed + SEED_OFFSET)
            base_mov = make_mlp(42 + seed + SEED_OFFSET)
            cal_mov = CalibratedClassifierCV(estimator=base_mov, cv=3, method='isotonic')
            cal_mov.fit(X_mov, y_mov)
            pm_list.append(cal_mov.predict_proba(X_vas)[:, 1])
            mov_cals.append(cal_mov)
        pm = np.mean(np.stack(pm_list, axis=0), axis=0)
        # === Stage-A: train LightGBM in parallel to MLP ===
        try:
            lgbm_A = make_lgbm_movement()
            X_train_A, y_train_A = X_mov, y_mov
            lgbm_A.fit(X_train_A, y_train_A)
            p_move_lgb = lgbm_A.predict_proba(X_vas)[:, 1] if hasattr(lgbm_A, 'predict_proba') else lgbm_A.predict(X_vas)
            if p_move_lgb is not None and len(p_move_lgb) == len(pm):
                pm = 0.5 * (pm + np.asarray(p_move_lgb).astype(np.float32))
        except Exception:
            pass
        pred_m = (pm >= 0.5).astype(int)
        # Trust check: movement with embargoed and future sentiment
        pm_list_emb = []
        pm_list_fut = []
        X_mov_emb, y_mov_emb = balance_binary(X_trs_emb, y_tr_m)
        X_mov_fut, y_mov_fut = balance_binary(X_trs_fut, y_tr_m)
        for seed in range(ENSEMBLE_SEEDS):
            np.random.seed(52 + seed + SEED_OFFSET)
            cal_mov_emb = CalibratedClassifierCV(estimator=make_mlp(52 + seed + SEED_OFFSET), cv=3, method='isotonic')
            cal_mov_emb.fit(X_mov_emb, y_mov_emb)
            pm_list_emb.append(cal_mov_emb.predict_proba(X_vas_emb)[:, 1])
            cal_mov_fut = CalibratedClassifierCV(estimator=make_mlp(152 + seed + SEED_OFFSET), cv=3, method='isotonic')
            cal_mov_fut.fit(X_mov_fut, y_mov_fut)
            pm_list_fut.append(cal_mov_fut.predict_proba(X_vas_fut)[:, 1])
        pm_emb = np.mean(np.stack(pm_list_emb, axis=0), axis=0)
        pm_fut = np.mean(np.stack(pm_list_fut, axis=0), axis=0)
    else:
        mov = CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=2000), cv=3, method='sigmoid')
        mov.fit(X_trs, y_tr_m)
        pred_m = mov.predict(X_vas)
    from sklearn.metrics import f1_score, accuracy_score
    print(f"StageA Movement | Acc={accuracy_score(y_va_m, pred_m):.3f} F1={f1_score(y_va_m, pred_m, average='macro'):.3f}")

    # Stage B: bull/bear on movement-only
    tr_mask = (y_tr != 2); va_mask = (y_va != 2)
    if USE_MLP:
        # Stage B ensemble
        X_dir_raw = X_trs[tr_mask]
        y_dir_raw = (y_tr[tr_mask] == 1).astype(np.int64)
        X_dir, y_dir = balance_binary(X_dir_raw, y_dir_raw)
        pb_list = []; pbear_list = []
        dir_cals = []
        for seed in range(ENSEMBLE_SEEDS):
            np.random.seed(100 + seed + SEED_OFFSET)
            base_dir = make_mlp(100 + seed + SEED_OFFSET)
            cal_dir = CalibratedClassifierCV(estimator=base_dir, cv=3, method='isotonic')
            cal_dir.fit(X_dir, y_dir)
            probs = cal_dir.predict_proba(X_vas[va_mask])
            pbear_list.append(probs[:, 0])
            pb_list.append(probs[:, 1])
            dir_cals.append(cal_dir)
        pbear_va = np.mean(np.stack(pbear_list, axis=0), axis=0)
        pb_va = np.mean(np.stack(pb_list, axis=0), axis=0)
        # === Stage-B: train LightGBM in parallel to MLP ===
        try:
            lgbm_B = make_lgbm_direction()
            X_train_B, y_train_B = X_dir, y_dir
            lgbm_B.fit(X_train_B, y_train_B)
            probs_lgb = lgbm_B.predict_proba(X_vas[va_mask]) if hasattr(lgbm_B, 'predict_proba') else None
            if probs_lgb is not None and probs_lgb.shape[0] == np.sum(va_mask):
                pb_lgb = probs_lgb[:, 1]
                # average bull prob; recompute bear as 1-pb
                m = min(len(pb_va), len(pb_lgb))
                if m > 0:
                    pb_va[:m] = 0.5 * (pb_va[:m] + pb_lgb[:m])
                    pbear_va[:m] = 1.0 - pb_va[:m]
        except Exception:
            pass
        pred_b = (pb_va >= 0.5).astype(int)
        # Trust check: direction with embargoed and future sentiment
        X_dir_emb, y_dir_emb = balance_binary(X_trs_emb[tr_mask], y_dir_raw)
        X_dir_fut, y_dir_fut = balance_binary(X_trs_fut[tr_mask], y_dir_raw)
        pb_list_emb = []; pbear_list_emb = []
        pb_list_fut = []; pbear_list_fut = []
        for seed in range(ENSEMBLE_SEEDS):
            np.random.seed(110 + seed + SEED_OFFSET)
            cal_dir_emb = CalibratedClassifierCV(estimator=make_mlp(110 + seed + SEED_OFFSET), cv=3, method='isotonic')
            cal_dir_emb.fit(X_dir_emb, y_dir_emb)
            probs_emb = cal_dir_emb.predict_proba(X_vas_emb[va_mask])
            pbear_list_emb.append(probs_emb[:, 0]); pb_list_emb.append(probs_emb[:, 1])
            cal_dir_fut = CalibratedClassifierCV(estimator=make_mlp(210 + seed + SEED_OFFSET), cv=3, method='isotonic')
            cal_dir_fut.fit(X_dir_fut, y_dir_fut)
            probs_fut = cal_dir_fut.predict_proba(X_vas_fut[va_mask])
            pbear_list_fut.append(probs_fut[:, 0]); pb_list_fut.append(probs_fut[:, 1])
        pbear_emb = np.mean(np.stack(pbear_list_emb, axis=0), axis=0)
        pb_emb = np.mean(np.stack(pb_list_emb, axis=0), axis=0)
        pbear_fut = np.mean(np.stack(pbear_list_fut, axis=0), axis=0)
        pb_fut = np.mean(np.stack(pb_list_fut, axis=0), axis=0)
    else:
        dirc = CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=2000), cv=3, method='sigmoid')
        dirc.fit(X_trs[tr_mask], (y_tr[tr_mask] == 1).astype(np.int64))
        pred_b = dirc.predict(X_vas[va_mask])
    print(f"StageB Direction | Acc={accuracy_score((y_va[va_mask]==1).astype(int), pred_b):.3f} F1={f1_score((y_va[va_mask]==1).astype(int), pred_b, average='macro'):.3f}")
    # Persist trained ensembles for deployment
    try:
        import joblib
        out_dir = BASE_DIR / 'hybrid_models'
        out_dir.mkdir(parents=True, exist_ok=True)
        if 'mov_cals' in locals() and len(mov_cals) > 0:
            joblib.dump(mov_cals, out_dir / 'stageA_movement_ensemble.pkl')
        if 'dir_cals' in locals() and len(dir_cals) > 0:
            joblib.dump(dir_cals, out_dir / 'stageB_direction_ensemble.pkl')
        print('Saved Stage A/B ensembles to', out_dir)
    except Exception as _e:
        print('Ensemble save failed:', _e)
    # Sentiment trust check diagnostics
    try:
        from sklearn.metrics import roc_auc_score
        y_va_m = (y_va != 2).astype(np.int64)
        print(f"Sentiment trust | AUC mov (base/emb/fut): {roc_auc_score(y_va_m, pm):.3f}/{roc_auc_score(y_va_m, pm_emb):.3f}/{roc_auc_score(y_va_m, pm_fut):.3f}")
        if 'pb_emb' in locals() and len(pbear_emb)==len(pb_emb)==np.sum(va_mask):
            y_dir_va = (y_va[va_mask]==1).astype(np.int64)
            print(f"Sentiment trust | AUC dir (base/emb/fut): {roc_auc_score(y_dir_va, pb_va):.3f}/{roc_auc_score(y_dir_va, pb_emb):.3f}/{roc_auc_score(y_dir_va, pb_fut):.3f}")
    except Exception as _e:
        print('Sentiment trust metrics skipped:', _e)

    # Per-asset overrides (train per-asset Stage A/B if enough samples)
    assets_all = np.array([s[5] for s in samples])
    a_tr = assets_all[tr_idx][tr_keep]
    a_va = assets_all[va_idx][va_keep]

    if USE_MLP:
        pm_asset = pm.copy()
        pbear_asset = pbear_va.copy() if 'pbear_va' in locals() else None
        pb_asset = pb_va.copy() if 'pb_va' in locals() else None
        uniq_assets = sorted(np.unique(a_va))
        for a in uniq_assets:
            tr_sel = (a_tr == a)
            va_sel = (a_va == a)
            if tr_sel.sum() < MIN_ASSET_TRAIN or va_sel.sum() == 0:
                continue
            X_tr_a = X_trs[tr_sel]
            y_tr_m_a = (y_tr[tr_sel] != 2).astype(np.int64)
            X_va_a = X_vas[va_sel]
            # Stage A
            X_mov_a, y_mov_a = balance_binary(X_tr_a, y_tr_m_a)
            pm_list_a = []
            for seed in range(ENSEMBLE_SEEDS):
                np.random.seed(200 + seed + SEED_OFFSET)
                base_mov = make_mlp(200 + seed + SEED_OFFSET)
                cal_mov = CalibratedClassifierCV(estimator=base_mov, cv=3, method='isotonic')
                cal_mov.fit(X_mov_a, y_mov_a)
                pm_list_a.append(cal_mov.predict_proba(X_va_a)[:, 1])
            pm_asset[va_sel] = np.mean(np.stack(pm_list_a, axis=0), axis=0)
            # Stage B
            tr_b_sel = tr_sel & (y_tr != 2)
            va_b_sel = va_sel & (y_va != 2)
            if tr_b_sel.sum() >= MIN_ASSET_TRAIN//2 and va_b_sel.sum() > 0:
                X_dir_a, y_dir_a = balance_binary(X_trs[tr_b_sel], (y_tr[tr_b_sel] == 1).astype(np.int64))
                probs_list = []
                for seed in range(ENSEMBLE_SEEDS):
                    np.random.seed(300 + seed + SEED_OFFSET)
                    base_dir = make_mlp(300 + seed + SEED_OFFSET)
                    cal_dir = CalibratedClassifierCV(estimator=base_dir, cv=3, method='isotonic')
                    cal_dir.fit(X_dir_a, y_dir_a)
                    probs_list.append(cal_dir.predict_proba(X_vas[va_b_sel]))
                probs_avg = np.mean(np.stack(probs_list, axis=0), axis=0)
                # map back into movement-only arrays
                mv_idx_a = np.where(va_b_sel)[0]
                m = min(len(mv_idx_a), probs_avg.shape[0])
                if m > 0:
                    if pbear_asset is not None and pb_asset is not None:
                        pbear_asset[:m] = probs_avg[:m, 0]
                        pb_asset[:m] = probs_avg[:m, 1]
        # apply per-asset overrides
        pm = pm_asset
        if pbear_asset is not None and pb_asset is not None:
            pbear_va = pbear_asset
            pb_va = pb_asset

    # Gated backtest
    # Use ensemble-averaged probabilities
    if not USE_MLP:
        # Fallback to single models if not MLP path
        if hasattr(mov, 'predict_proba'):
            pm = mov.predict_proba(X_vas)[:,1]
        else:
            df = mov.decision_function(X_vas)
            pm = 1/(1+np.exp(-df)) if df.ndim==1 else 1/(1+np.exp(-df[:,1]))
        # Direction probabilities only defined for movement samples
        mv_idx = np.where(y_va != 2)[0]
        probs_dir = dirc.predict_proba(X_vas[mv_idx]) if len(mv_idx) > 0 else np.zeros((0,2), dtype=np.float32)
        pb_full = np.zeros(len(pm), dtype=np.float32)
        pbear_full = np.zeros(len(pm), dtype=np.float32)
        if len(mv_idx) > 0:
            pb_full[mv_idx] = probs_dir[:,1]
            pbear_full[mv_idx] = probs_dir[:,0]
        pb = pb_full
        pbear = pbear_full
    else:
        # We have ensemble-averaged pb_va/pbear_va computed on movement-only mask
        mv_idx = np.where(y_va != 2)[0]
        pb_full = np.zeros(len(pm), dtype=np.float32)
        pbear_full = np.zeros(len(pm), dtype=np.float32)
        # Ensure lengths align (defensive trim)
        m = min(len(mv_idx), len(pb_va), len(pbear_va))
        if m > 0:
            pb_full[mv_idx[:m]] = pb_va[:m]
            pbear_full[mv_idx[:m]] = pbear_va[:m]
        pb = pb_full
        pbear = pbear_full

    # Optional smoothing
    if SMOOTH_ALPHA > 0.0 and len(pm) > 1:
        def ema(arr, a):
            out = np.zeros_like(arr)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = a * arr[i] + (1 - a) * out[i-1]
            return out
        pm = ema(pm, SMOOTH_ALPHA)
        margin_vec = np.abs(pb - pbear)
        margin_vec = ema(margin_vec, SMOOTH_ALPHA)
    else:
        margin_vec = np.abs(pb - pbear)

    # Per-asset gating with SL/TP clipping and confidence-sized positions
    intervals_va = iv_va
    assets_va = a_va
    uniq_assets = sorted(np.unique(assets_va))

    # Optional per-asset costs via env JSON (e.g., {"ADAUSD":0.0015,"AVAXUSD":0.001})
    try:
        _fees_env = os.environ.get('PER_ASSET_FEES', '')
        PER_ASSET_FEES = json.loads(_fees_env) if _fees_env.strip() else {}
    except Exception:
        PER_ASSET_FEES = {}
    try:
        _spreads_env = os.environ.get('PER_ASSET_SPREADS', '')
        PER_ASSET_SPREADS = json.loads(_spreads_env) if _spreads_env.strip() else {}
    except Exception:
        PER_ASSET_SPREADS = {}

    def compute_stats_for_indices(idxs: np.ndarray, m_conf: float, diff: float):
        if len(idxs) == 0:
            return None
        idxs = np.array(sorted(idxs, key=lambda i: t_va[i]))
        pnl = []
        trades = 0
        avg_size = 0.0
        month_counts: Dict[tuple, int] = {}
        last_dir = 0
        cooldown = 0
        for i in idxs:
            if len(TRADE_ONLY_INTERVALS) > 0 and intervals_va[i] not in TRADE_ONLY_INTERVALS:
                continue
            ts = pd.Timestamp(t_va[i])
            mkey = (ts.year, ts.month)
            if month_counts.get(mkey, 0) >= TRADES_PER_MONTH_CAP:
                continue
            if cooldown > 0:
                cooldown -= 1
                continue
            if pm[i] < m_conf:
                continue
            margin = abs(pb[i] - pbear[i])
            # Regime-aware diff adjustment
            macd_i = float(ts_va_feats[i, 4]) if ts_va_feats.shape[1] >= 5 else 0.0
            rsi_i = float(ts_va_feats[i, 5]) if ts_va_feats.shape[1] >= 6 else 0.0
            is_trending = (abs(macd_i) + abs(rsi_i)) >= trend_thr
            # Optional ADX filter in trending conditions
            if is_trending and adx_col is not None and SKIP_LOW_ADX_TREND:
                adx_i = float(ts_va_feats[i, adx_col])
                if adx_i < ADX_MIN_TREND:
                    continue
            diff_eff = diff
            if is_trending:
                diff_eff = max(DIFF_MIN, diff - TREND_DIFF_REDUCTION)
            else:
                diff_eff = min(DIFF_MAX, diff + CHOP_DIFF_ADD)
                # In squeeze (very low BB bandwidth), be extra conservative in chop
                if bb_col is not None and bb_squeeze_thr is not None:
                    bb_i = float(ts_va_feats[i, bb_col])
                    if bb_i <= bb_squeeze_thr:
                        diff_eff = min(DIFF_MAX, diff_eff + CHOP_DIFF_ADD_EXTRA)
            if margin < diff_eff:
                continue
            cur_dir = 1 if pb[i] > pbear[i] else -1
            if REQUIRE_TREND_ALIGN and is_trending:
                trend_dir = 1 if macd_i >= 0 else -1
                if cur_dir != trend_dir:
                    continue
            if cur_dir < 0 and margin < (diff + SHORT_MARGIN_EXTRA):
                continue
            if last_dir != 0 and cur_dir != last_dir and margin < (diff + WHIP_MARGIN_BOOST):
                continue
            # Position sizing via capped Kelly on directional probs
            try:
                edge = float(pb[i] - pbear[i])
                kelly_raw = edge / max(1e-6, 1.0 - abs(edge))
                kelly_scaled = kelly_raw * float(CONFIG['execution']['kelly_scale'])
                size_cap_eff = min(SIZE_CAP, float(CONFIG['execution']['size_cap'])) if 'execution' in CONFIG else SIZE_CAP
                size_mag = max(SIZE_FLOOR, min(abs(kelly_scaled), size_cap_eff))
                size = size_mag
            except Exception:
                # fallback to legacy sizing
                size = max(SIZE_FLOOR, (margin - diff) / (1.0 - diff))
                size *= min(1.0, VOL_INV_K / (ts_std_va[i] + 1e-8))
                size *= (LONG_SIZE_MULT if cur_dir > 0 else SHORT_SIZE_MULT)
                size = max(SIZE_FLOOR, min(size, SIZE_CAP))

            # ATR-based TP/SL distances (fallback to sigma-based if ATR unavailable)
            sigma = ts_std_va[i]
            tp_mult_cfg = float(CONFIG['execution']['tp_sl']['atr_mult_tp']) if 'execution' in CONFIG else TP_MULT
            sl_mult_cfg = float(CONFIG['execution']['tp_sl']['atr_mult_sl']) if 'execution' in CONFIG else SL_MULT
            atr_now = None
            entry_price = fwd_entry_price[i] if i < len(fwd_entry_price) else None
            try:
                wp = window_paths[i] if i < len(window_paths) else None
                if wp is not None and Path(wp).exists():
                    df_win = pd.read_pickle(wp)
                    atr_series = compute_atr14(df_win)
                    if atr_series is not None and len(atr_series) > 0 and not np.isnan(atr_series.iloc[-1]):
                        atr_now = float(atr_series.iloc[-1])
                if entry_price is None and wp is not None and Path(wp).exists():
                    df_win = pd.read_pickle(wp)
                    entry_price = float(df_win['close'].iloc[-1]) if 'close' in df_win.columns else None
            except Exception:
                pass
            if atr_now is not None and entry_price is not None and entry_price > 0:
                tp_ret = (tp_mult_cfg * atr_now) / entry_price
                sl_ret = (sl_mult_cfg * atr_now) / entry_price
            else:
                # fallback to sigma multiples
                if is_trending:
                    tp_mult_local = TP_MULT * (1.0 + TP_TREND_BOOST)
                    sl_mult_local = SL_MULT * SL_TREND_REL
                else:
                    tp_mult_local = max(0.25, TP_MULT * (1.0 - TP_CHOP_CUT))
                    sl_mult_local = SL_MULT * SL_CHOP_REL
                tp_ret = tp_mult_local * sigma
                sl_ret = sl_mult_local * sigma
            # Compute executed return using forward path if available
            if fwd_returns[i] is not None and isinstance(fwd_returns[i], np.ndarray) and len(fwd_returns[i]) > 0:
                ret_path = fwd_returns[i]
                # Directional returns path
                path_dir = ret_path if cur_dir > 0 else (-ret_path)
                exit_idx = len(path_dir) - 1
                # Trailing stop based on sigma
                if USE_TRAIL_STOP and len(path_dir) > 1:
                    cummax = np.maximum.accumulate(path_dir)
                    dd = cummax - path_dir
                    thr = TRAIL_MULT * sigma
                    hit = np.where(dd >= thr)[0]
                    if len(hit) > 0:
                        exit_idx = int(hit[0])
                # Time stop
                if USE_TIME_STOP:
                    ts_idx = min(TIME_STOP_BARS - 1, len(path_dir) - 1)
                    exit_idx = min(exit_idx, ts_idx)
                signed_ret = float(path_dir[exit_idx])
                clipped = min(max(signed_ret, -sl_ret), tp_ret)
            else:
                signed_ret = float(c_va[i]) if cur_dir > 0 else float(-c_va[i])
                clipped = min(max(signed_ret, -sl_ret), tp_ret)
            # Per-asset fee/spread if provided
            a_i = str(assets_va[i]) if i < len(assets_va) else ''
            fee_loc = float(PER_ASSET_FEES.get(a_i, FEE))
            spread_loc = float(PER_ASSET_SPREADS.get(a_i, SPREAD))
            trade_pnl = size * (clipped - fee_loc) - spread_loc
            pnl.append(trade_pnl)
            trades += 1
            avg_size += size
            month_counts[mkey] = month_counts.get(mkey, 0) + 1
            if trade_pnl < 0.0:
                cooldown = COOLDOWN_BARS
            last_dir = cur_dir
        if len(pnl) == 0:
            return None
        pnl = np.array(pnl, dtype=np.float32)
        eq = np.cumsum(pnl)
        roll_max = np.maximum.accumulate(eq)
        dd = roll_max - eq
        max_dd = float(dd.max()) if len(dd) else 0.0
        sharpe = float(pnl.mean() / (pnl.std() + 1e-8))
        total = float(eq[-1])
        stat = {'m_conf': m_conf, 'diff': diff, 'pnl': total, 'sharpe': sharpe, 'max_dd': max_dd, 'trades': trades, 'avg_size': (avg_size / max(trades,1))}
        if trades >= MIN_TRADES_PER_ASSET and max_dd <= MAX_DD:
            return stat
        return None

    def evaluate_slice(idxs: np.ndarray):
        if len(idxs) == 0:
            return None
        # sort by time to apply cooldown/whipsaw in order
        idxs = np.array(sorted(idxs, key=lambda i: t_va[i]))
        pm_q_val = np.quantile(pm[idxs], PM_Q)
        margin_local = np.abs(pb[idxs] - pbear[idxs])
        margin_q_val = np.quantile(margin_local, MARGIN_Q)
        # include Optuna overrides and wider exploration
        m_grid = [
            max(MCONF_MIN, pm_q_val - 0.10),
            pm_q_val,
            min(MCONF_MAX, pm_q_val + 0.10),
        ]
        if USE_OVR_GATES and 'm_conf' in OVR:
            m_grid.append(float(OVR['m_conf']))
        m_grid = sorted(set([float(np.clip(x, MCONF_MIN, MCONF_MAX)) for x in m_grid]))

        d_grid = [
            max(DIFF_MIN, margin_q_val - 0.05),
            margin_q_val,
            min(DIFF_MAX, margin_q_val + 0.05),
        ]
        if USE_OVR_GATES and 'diff' in OVR:
            d_grid.append(float(OVR['diff']))
        # also probe a very small diff
        d_grid.append(max(DIFF_MIN, 0.01))
        d_grid = sorted(set([float(np.clip(x, DIFF_MIN, DIFF_MAX)) for x in d_grid]))
        best = None
        best_wf = -999.0
        for m_conf in m_grid:
            for diff in d_grid:
                # compute base stats
                stat = compute_stats_for_indices(idxs, m_conf, diff)
                # stability check with jittered gates
                if stat is not None and (STAB_MCONF_JITTER > 0 or STAB_DIFF_JITTER > 0):
                    stat_j = compute_stats_for_indices(idxs, max(MCONF_MIN, min(MCONF_MAX, m_conf + STAB_MCONF_JITTER)), max(DIFF_MIN, min(DIFF_MAX, diff + STAB_DIFF_JITTER)))
                    stat_k = compute_stats_for_indices(idxs, max(MCONF_MIN, min(MCONF_MAX, m_conf - STAB_MCONF_JITTER)), max(DIFF_MIN, min(DIFF_MAX, diff - STAB_DIFF_JITTER)))
                    sharpe_min = min([s['sharpe'] for s in [stat, stat_j, stat_k] if s is not None]) if (stat_j or stat_k) else stat['sharpe']
                    if sharpe_min < STAB_SHARPE_MIN:
                        stat = None
                if stat is not None:
                    if USE_WF_GRID_SELECT:
                        # evaluate WF Sharpe for this (m_conf, diff)
                        try:
                            folds_local = max(2, WF_FOLDS)
                            n_va = len(y_va)
                            fold_sz = max(1, n_va // folds_local)
                            sh_list = []; trades_total = 0
                            for k in range(folds_local):
                                start = k * fold_sz
                                end = n_va if k == folds_local - 1 else (k + 1) * fold_sz
                                fold_idxs = idxs[(idxs >= start) & (idxs < end)]
                                stat_k = compute_stats_for_indices(fold_idxs, m_conf, diff)
                                if stat_k is not None:
                                    sh_list.append(stat_k['sharpe']); trades_total += stat_k['trades']
                            wf_stat = float(np.median(sh_list)) if (WF_USE_MEDIAN and sh_list) else (float(np.mean(sh_list)) if sh_list else -999.0)
                            if trades_total < MIN_TRADES_PER_FOLD * folds_local:
                                wf_stat = -999.0
                        except Exception:
                            wf_stat = -999.0
                        # Reject if WF Sharpe below stability threshold
                        if wf_stat < STAB_WF_MIN_SHARPE:
                            continue
                        if wf_stat > best_wf or (abs(wf_stat - best_wf) < 1e-6 and stat['sharpe'] > (best['sharpe'] if best else -999.0)):
                            best = stat; best_wf = wf_stat
                    else:
                        if best is None or stat['sharpe'] > best['sharpe'] or (abs(stat['sharpe'] - best['sharpe']) < 1e-6 and stat['pnl'] > best['pnl']):
                            best = stat
        # Fallback: relaxed constraints and wider grids if nothing found
        if best is None:
            for m_conf in GRID_MCONF:
                for diff in (GRID_DIFF + [0.01, 0.02, 0.03]):
                    stat = compute_stats_for_indices(idxs, m_conf, diff)
                    if stat is not None:
                        if best is None or stat['sharpe'] > best['sharpe'] or (abs(stat['sharpe'] - best['sharpe']) < 1e-6 and stat['pnl'] > best['pnl']):
                            best = stat
        return best

    agg_pnl = 0.0; agg_trades = 0; agg_avg_size_num = 0.0; sharpes = []
    picks = []
    for a in uniq_assets:
        idxs = np.where(assets_va == a)[0]
        # Simple per-asset TP/SL grid search around global defaults
        best = None
        # Include any per-asset tp/sl overrides discovered by Optuna
        tp_candidates = [TP_MULT, max(0.5, TP_MULT*0.75), TP_MULT*1.25]
        sl_candidates = [SL_MULT, max(0.5, SL_MULT*0.75), SL_MULT*1.25]
        try:
            if PER_ASSET_OVR_JSON.exists():
                import json as _json
                _ovr = _json.loads(PER_ASSET_OVR_JSON.read_text())
                if a in _ovr:
                    v = _ovr[a]
                    if 'TP_MULT' in v:
                        tp_candidates.append(float(v['TP_MULT']))
                    if 'SL_MULT' in v:
                        sl_candidates.append(float(v['SL_MULT']))
        except Exception:
            pass
        for tp_mult in sorted(set(tp_candidates)):
            for sl_mult in sorted(set(sl_candidates)):
                # monkey-patch local TP/SL inside closure via globals
                old_tp, old_sl = globals()['TP_MULT'], globals()['SL_MULT']
                globals()['TP_MULT'], globals()['SL_MULT'] = tp_mult, sl_mult
                cand = evaluate_slice(idxs)
                globals()['TP_MULT'], globals()['SL_MULT'] = old_tp, old_sl
                if cand is None:
                    continue
                # track best by Sharpe, then PnL
                if best is None or cand['sharpe'] > best['sharpe'] or (abs(cand['sharpe']-best['sharpe'])<1e-6 and cand['pnl']>best['pnl']):
                    best = {**cand, 'tp_mult': tp_mult, 'sl_mult': sl_mult}
        if best is not None and best['trades'] >= MIN_TRADES_PER_ASSET:
            picks.append((a, best))
            agg_pnl += best['pnl']; agg_trades += best['trades']; agg_avg_size_num += best['avg_size'] * best['trades']
            sharpes.append(best['sharpe'])

    if agg_trades > 0:
        print("Per-asset gates:")
        for a, st in picks:
            tpsl_txt = ''
            if 'tp_mult' in st and 'sl_mult' in st:
                tpsl_txt = f" tp={st['tp_mult']:.2f} sl={st['sl_mult']:.2f}"
            print(f"  {a}: m_conf={st['m_conf']:.2f} diff={st['diff']:.2f} trades={st['trades']} sharpe={st['sharpe']:.2f} pnl={st['pnl']:.4f}{tpsl_txt}")
        # Compute per-asset walk-forward Sharpe for fixed gates and filter by stability
        asset_wf = {}
        try:
            folds_local = max(2, WF_FOLDS)
            n_va = len(y_va)
            fold_sz = max(1, n_va // folds_local)
            for a, st in picks:
                idxs_a = np.where(assets_va == a)[0]
                sh_list = []
                old_tp, old_sl = globals()['TP_MULT'], globals()['SL_MULT']
                globals()['TP_MULT'], globals()['SL_MULT'] = st.get('tp_mult', TP_MULT), st.get('sl_mult', SL_MULT)
                for k in range(folds_local):
                    start = k * fold_sz
                    end = n_va if k == folds_local - 1 else (k + 1) * fold_sz
                    fold_idxs = idxs_a[(idxs_a >= start) & (idxs_a < end)]
                    stat_k = compute_stats_for_indices(fold_idxs, float(st['m_conf']), float(st['diff']))
                    if stat_k is not None:
                        sh_list.append(stat_k['sharpe'])
                globals()['TP_MULT'], globals()['SL_MULT'] = old_tp, old_sl
                asset_wf[a] = float(np.mean(sh_list)) if sh_list else -999.0
        except Exception as _e:
            print('Per-asset WF Sharpe calc failed:', _e)

        # Apply selection filters
        picks_f = [(a, st) for (a, st) in picks
                   if st['sharpe'] >= SELECT_SHARPE_MIN and st['pnl'] >= SELECT_PNL_MIN
                   and st['trades'] >= SELECT_MIN_TRADES and st['max_dd'] <= SELECT_MAX_DD
                   and asset_wf.get(a, 0.0) >= STAB_WF_MIN_SHARPE]
        if len(picks_f) == 0:
            picks_f = picks  # fallback to all
        if SELECT_TOP_N and len(picks_f) > SELECT_TOP_N:
            if SELECT_TOP_BY == 'sharpe':
                picks_f = sorted(picks_f, key=lambda x: x[1]['sharpe'], reverse=True)[:SELECT_TOP_N]
            elif SELECT_TOP_BY == 'pnl':
                picks_f = sorted(picks_f, key=lambda x: x[1]['pnl'], reverse=True)[:SELECT_TOP_N]
            else:
                # composite: normalize sharpe and pnl to [0,1] then combine
                sh = np.array([st['sharpe'] for _, st in picks_f], dtype=np.float32)
                pn = np.array([st['pnl'] for _, st in picks_f], dtype=np.float32)
                tr = np.array([st['trades'] for _, st in picks_f], dtype=np.float32)
                def norm(arr):
                    lo, hi = float(np.min(arr)), float(np.max(arr))
                    return (arr - lo) / (hi - lo + 1e-8)
                sh_n = norm(sh); pn_n = norm(pn); tr_n = norm(tr)
                scores = sh_n + SELECT_ALPHA * pn_n + SELECT_BETA_TRADES * tr_n
                ranked = sorted(zip(picks_f, scores), key=lambda t: float(t[1]), reverse=True)
                picks_f = [pf for (pf, _) in ranked[:SELECT_TOP_N]]
        agg_pnl_f = sum(st['pnl'] for _, st in picks_f)
        agg_trades_f = sum(st['trades'] for _, st in picks_f)
        agg_avg_size_num_f = sum(st['avg_size']*st['trades'] for _, st in picks_f)
        sharpes_f = [st['sharpe'] for _, st in picks_f]
        print("Selected assets:")
        print(" ".join(a for a, _ in picks_f))
        print(f"Aggregate Backtest (val): PnL={agg_pnl_f:.4f} MeanSharpe={np.mean(sharpes_f):.2f} Trades={agg_trades_f} AvgSize={(agg_avg_size_num_f/max(1,agg_trades_f)):.3f}")
        # Save selected per-asset gates and summary for reuse
        try:
            out = BASE_DIR / 'hybrid_models' / 'per_asset_gates.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            gates = {a: {'m_conf': float(st['m_conf']), 'diff': float(st['diff']), 'trades': int(st['trades']), 'sharpe': float(st['sharpe']), 'pnl': float(st['pnl']), 'max_dd': float(st['max_dd']), 'tp_mult': float(st.get('tp_mult', TP_MULT)), 'sl_mult': float(st.get('sl_mult', SL_MULT))} for a, st in picks_f}
            with open(out, 'w') as f:
                json.dump(gates, f)
            print('Saved per-asset gates to', out)
            # Write a brief summary file
            summary = {
                'selected_assets': [a for a, _ in picks_f],
                'aggregate': {'pnl': float(agg_pnl_f), 'mean_sharpe': float(np.mean(sharpes_f)), 'trades': int(agg_trades_f)},
                'env': {
                    'USE_OVR_GATES': USE_OVR_GATES, 'WF_FOLDS': WF_FOLDS,
                    'ADX_MIN_TREND': ADX_MIN_TREND, 'BB_SQUEEZE_Q': BB_SQUEEZE_Q,
                    'STAB_MCONF_JITTER': STAB_MCONF_JITTER, 'STAB_DIFF_JITTER': STAB_DIFF_JITTER,
                    'SELECT_TOP_N': SELECT_TOP_N, 'SELECT_TOP_BY': SELECT_TOP_BY,
                    'SELECT_ALPHA': SELECT_ALPHA, 'SELECT_BETA_TRADES': SELECT_BETA_TRADES,
                }
            }
            with open(BASE_DIR / 'hybrid_models' / 'summary_latest.json', 'w') as f:
                json.dump(summary, f)
            # Save feature schema (sizes and external keys) for deployment
            schema = {
                'blocks': {
                    'dino_dim': int(Xd.shape[1]),
                    'ts_dim': int(Xt.shape[1]),
                    'yolo_dim': int(Xy.shape[1]),
                    'ctx_dim': int(Xc.shape[1]),
                    'ext_dim': int(Xe.shape[1]),
                    'sent_dim': int(Xs_emb.shape[1]),
                },
                'external_keys': get_external_feature_keys(samples, meta),
                'embed_model_id': EMBED_MODEL_ID,
                'active_embed_dir': str(ACTIVE_EMBED_DIR),
            }
            with open(BASE_DIR / 'hybrid_models' / 'feature_schema.json', 'w') as f:
                json.dump(schema, f)
        except Exception as _e:
            print('Save per-asset gates failed:', _e)
        # Simple walk-forward evaluation (3 folds) on temporal order
        try:
            folds = max(2, WF_FOLDS)
            n_va = len(y_va)
            fold_sz = max(1, n_va // folds)
            wf_stats = []
            for k in range(folds):
                start = k * fold_sz
                end = n_va if k == folds - 1 else (k + 1) * fold_sz
                idxs = np.arange(start, end)
                # reuse evaluate_slice to get stats on this slice
                best_fold = evaluate_slice(idxs)
                if best_fold is not None:
                    wf_stats.append(best_fold['sharpe'])
            if wf_stats:
                print(f"Walk-forward Sharpe (mean/median over {folds} folds): {np.mean(wf_stats):.2f}/{np.median(wf_stats):.2f}")
        except Exception as _e:
            print('Walk-forward eval failed:', _e)
        # Fee sensitivity analysis (optional)
        if FEE_SENSITIVITY.strip():
            try:
                fees = [float(x.strip()) for x in FEE_SENSITIVITY.split(',') if x.strip()]
                for fee_val in fees:
                    old_fee = globals()['FEE']
                    globals()['FEE'] = fee_val
                    agg_pnl_s = 0.0; agg_tr_s = 0; agg_sz_s = 0.0; sh_list_s = []
                    picks_s = []
                    for a in uniq_assets:
                        idxs = np.where(assets_va == a)[0]
                        best = None
                        for tp_mult in [TP_MULT, max(0.5, TP_MULT*0.75), TP_MULT*1.25]:
                            for sl_mult in [SL_MULT, max(0.5, SL_MULT*0.75), SL_MULT*1.25]:
                                old_tp, old_sl = globals()['TP_MULT'], globals()['SL_MULT']
                                globals()['TP_MULT'], globals()['SL_MULT'] = tp_mult, sl_mult
                                cand = evaluate_slice(idxs)
                                globals()['TP_MULT'], globals()['SL_MULT'] = old_tp, old_sl
                                if cand is None:
                                    continue
                                if best is None or cand['sharpe'] > best['sharpe'] or (abs(cand['sharpe']-best['sharpe'])<1e-6 and cand['pnl']>best['pnl']):
                                    best = cand
                        if best is not None and best['trades'] >= MIN_TRADES_PER_ASSET:
                            picks_s.append((a, best))
                            agg_pnl_s += best['pnl']; agg_tr_s += best['trades']; agg_sz_s += best['avg_size'] * best['trades']
                            sh_list_s.append(best['sharpe'])
                    globals()['FEE'] = old_fee
                    if agg_tr_s > 0:
                        print(f"Fee sensitivity (fee={fee_val:.4f}) -> PnL={agg_pnl_s:.4f} MeanSharpe={np.mean(sh_list_s):.2f} Trades={agg_tr_s} AvgSize={(agg_sz_s/max(1,agg_tr_s)):.3f}")
            except Exception as _e:
                print('Fee sensitivity analysis failed:', _e)
    else:
        # Global fallback across all assets combined
        all_idxs = np.arange(len(assets_va))
        best_global = evaluate_slice(all_idxs)
        if best_global is not None:
            st = best_global
            print("Global fallback gates:")
            print(f"  m_conf={st['m_conf']:.2f} diff={st['diff']:.2f} trades={st['trades']} sharpe={st['sharpe']:.2f} pnl={st['pnl']:.4f}")
        else:
            print('Two-stage backtest produced no trades.')


if __name__ == '__main__':
    main()
    # Minimal run summary (best-effort; relies on printed stats above)
    try:
        print("\n=== RUN SUMMARY ===")
        # Metrics are printed inline during training/backtest; here we indicate completion
        print("Done.")
    except Exception:
        pass


