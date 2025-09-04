"""
Part 0: Setup and Data Generation for Hybrid Crypto Trading System

Capabilities:
- Load Kraken historical 1m OHLCV from local CSV (fallback: yfinance; optional CCXT append)
- Generate multi-interval data (1m,5m,15m,60m,240m,1440m) with robust gap handling
- Compute ICT indicators (swing_highs_lows, fvg, ob, bos_choch) and preserve all logic
- Sliding-window chart generation with overlays and balanced labels via quantiles
- Save per-class images, corresponding window pickles, and rich metadata
- Create YOLOv8 pseudo-labels for ICT patterns (normalized bboxes) and optional model inference
- Precompute DinoV2 embeddings for all images for fast downstream training
- Fetch and align external macro/derivatives/on-chain context (DXY proxy, SPX, VIX, Gold, Fear&Greed, Binance funding; optional MVRV Z) to 4H and enrich metadata

Notes:
- This script is OS-agnostic and uses absolute paths under the user's home directory by default.
- TA-Lib is optional. If not installed, indicators are computed with pandas/numpy.
- Designed to run locally on CPU or GPU. Install requirements are handled at runtime.
"""

import os
import sys
import json
import math
import time
import shutil
import random
import zipfile
import warnings
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================
# Part0 v2: Config, seeds, workspace
# ============================
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore
try:
    from utils.seed import set_all_seeds  # type: ignore
except Exception:
    def set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def load_config(path: str = "conf/experiment.yaml") -> dict:
    if path and os.path.exists(path) and yaml is not None:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            logging.exception(f"Failed to load config from {path}, using defaults")
            pass
    return {
        "data": {
            "symbols": ["ADAUSD","AVAXUSD","LINKUSD","LTCUSD","SOLUSD","XRPUSD"],
            "interval": "1h",
            "lookback_bars": 240,
            "horizon_bars": 12,
        },
        "features": {
            "microstructure": True,
            "volatility_har": True,
            "regime": True,
            "funding_basis": True,
            "dino_embeddings": False,
            "ict_counts": False,
        },
        "training": {"folds": 5, "embargo_bars": 10},
    }


CONF = load_config()
set_all_seeds(42)

# Workspace (v2)
WS = Path(os.environ.get("HYBRID_WORKSPACE", "/content/hybrid_workspace"))
DS = WS / "datasets" / "chart_dataset"
DS.mkdir(parents=True, exist_ok=True)
CACHE = WS / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Install/Ensure Dependencies
# ----------------------------
def ensure(pkg: str) -> None:
    try:
        __import__(pkg.split('==')[0].split('[')[0])
    except Exception:
        import subprocess
        import logging
        logging.exception(f"Failed to import {pkg}, attempting pip install")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])


for req in [
    'numpy',
    'pandas',
    'matplotlib',
    'mplfinance',
    'Pillow',
    'ccxt',
    'yfinance',
    'requests',
    'scikit-learn',
    'ultralytics',
    'transformers',
    'torch',
    'torchvision',
]:
    ensure(req)

import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

import yfinance as yf
import ccxt
import requests

import torch
from transformers import AutoImageProcessor, Dinov2Model


# ----------------------------
# Colab detection and Drive (optional)
# ----------------------------
def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def maybe_mount_drive() -> None:
    if not in_colab():
        return
    use_drive = os.environ.get('USE_DRIVE', '1')  # default to mount in Colab
    if use_drive != '1':
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
        print('Drive mounted at /content/drive')
    except Exception as e:
        print(f'Google Drive mount skipped: {e}')


HOME_DIR = Path.home()

# Prefer Colab-friendly defaults under /content, else fall back to ~/Documents
if in_colab():
    maybe_mount_drive()
    default_base = Path('/content/hybrid_workspace')
    default_kraken_dir = Path('/content/kraken_data')
    default_zip = Path('/content/drive/MyDrive/Kraken_OHLCVT.zip')
else:
    default_base = HOME_DIR / 'Documents' / 'crypto_hybrid'
    default_kraken_dir = HOME_DIR / 'Documents' / 'kraken_data'
    default_zip = HOME_DIR / 'Documents' / 'Kraken_OHLCVT.zip'

BASE_DIR = Path(os.environ.get('HYBRID_BASE_DIR', str(WS)))
DATASET_DIR = DS
WINDOWS_DIR = DATASET_DIR / 'windows'
WINDOWS_FWD_DIR = DATASET_DIR / 'windows_fwd'
EMBED_DIR = DATASET_DIR / 'embeddings_dino'
YOLO_DET_DIR = BASE_DIR / 'datasets' / 'yolo_ict_det'
EXTERNAL_DIR = BASE_DIR / 'datasets' / 'external'

# Where to search for Kraken CSVs (unzipped) like BTCUSD_1.csv
KRAKEN_DATA_DIR = Path(os.environ.get('KRAKEN_DATA_DIR', str(default_kraken_dir)))

# Optional: Zip to extract if directory is empty
KRAKEN_ZIP_PATH = Path(os.environ.get('KRAKEN_ZIP_PATH', str(default_zip)))

ASSETS: List[str] = [
    'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD',
    'LTCUSD', 'XRPUSD', 'AVAXUSD', 'LINKUSD',
    'DOTUSD', 'ATOMUSD', 'SUIUSD', 'UNIUSD'
]

# Minutes per bar for multi-interval generation (focus on 4h and 1d)
INTERVALS_MIN: List[int] = [240, 1440]

# Sliding window config
SEQ_LEN: int = 50
STEP: int = 10
FORECAST_HORIZON_BARS: int = 20

# Per asset-interval cap to control dataset size
MAX_WINDOWS_PER_COMBO: int = 2000

# Plotting
FIGSIZE = (6, 4)
DPI = 200

# YOLO detection pseudo-label classes (ICT-focused)
YOLO_CLASSES: List[str] = [
    'fvg_bull', 'fvg_bear',
    'ob_bull', 'ob_bear',
    'bos_bull', 'bos_bear',
    'choch_bull', 'choch_bear'
]

# Optional: path to a trained YOLOv8 detection model for pattern detection on charts
YOLO_DET_MODEL_PATH: Optional[str] = os.environ.get('YOLO_ICT_MODEL_PATH', None)

# yfinance fallback mapping
ASSET_TO_YF: Dict[str, str] = {
    'XBTUSD': 'BTC-USD',
    'ETHUSD': 'ETH-USD',
    'SOLUSD': 'SOL-USD',
    'ADAUSD': 'ADA-USD',
    'LTCUSD': 'LTC-USD',
    'XRPUSD': 'XRP-USD',
    'AVAXUSD': 'AVAX-USD',
    'LINKUSD': 'LINK-USD',
    'DOTUSD': 'DOT-USD',
    'ATOMUSD': 'ATOM-USD',
    'SUIUSD': 'SUI-USD',
    'UNIUSD': 'UNI-USD',
}


# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def try_extract_zip_if_needed() -> None:
    if KRAKEN_DATA_DIR.exists() and any(KRAKEN_DATA_DIR.rglob('*.csv')):
        return
    if KRAKEN_ZIP_PATH.exists():
        safe_mkdir(KRAKEN_DATA_DIR)
        try:
            with zipfile.ZipFile(str(KRAKEN_ZIP_PATH), 'r') as zf:
                zf.extractall(str(KRAKEN_DATA_DIR))
            print(f"Extracted Kraken ZIP to {KRAKEN_DATA_DIR}")
        except Exception as e:
            print(f"ZIP extract skipped: {e}")


def find_csv_for_asset(asset: str) -> Optional[Path]:
    target = f"{asset}_1.csv"
    for p in KRAKEN_DATA_DIR.rglob(target):
        return p
    return None
def minutes_to_pandas_freq(minutes: int) -> str:
    # Prefer clear aliases to avoid month confusion (e.g., 'm')
    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days}D"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours}H"
    return f"{minutes}T"  # minutes



def robust_to_datetime_index(df: pd.DataFrame, col: str, unit: str = 's') -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], unit=unit)
    df.set_index(col, inplace=True)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df


# ----------------------------
# External data: macro/derivatives/on-chain
# ----------------------------
def fetch_macro_4h() -> pd.DataFrame:
    """
    Fetch macro context via yfinance (daily), then resample/ffill to 4H and add simple transforms.
    Symbols: UUP (DXY proxy), ^GSPC (SPX), ^VIX (VIX), GC=F (Gold).
    """
    tickers = {
        'UUP': 'dxy_proxy',
        '^GSPC': 'spx',
        '^VIX': 'vix',
        'GC=F': 'gold',
    }
    frames: List[pd.DataFrame] = []
    for yf_t, colname in tickers.items():
        try:
            df_yf = yf.download(yf_t, interval='1d', period='max', auto_adjust=False,
                                progress=False, prepost=False, threads=False, group_by='column')
            if df_yf is None or df_yf.empty:
                continue
            # Flatten possible MultiIndex columns and robustly select Close
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = ['_'.join([str(x) for x in c if x]) for c in df_yf.columns]
            # Find a Close column case-insensitively
            close_col = None
            for c in df_yf.columns:
                if str(c).lower() == 'close':
                    close_col = c
                    break
            if close_col is None:
                # Some tickers may return 'Adj Close' only; fall back to that
                for c in df_yf.columns:
                    if str(c).lower().replace(' ', '') == 'adjclose':
                        close_col = c
                        break
            if close_col is None:
                continue
            df_yf = df_yf[[close_col]].rename(columns={close_col: colname})
            df_yf.index = pd.to_datetime(df_yf.index)
            try:
                df_yf.index = df_yf.index.tz_localize(None)
            except Exception:
                pass
            frames.append(df_yf)
        except Exception as e:
            print(f"Macro fetch failed for {yf_t}: {e}")
    if not frames:
        return pd.DataFrame()
    daily = pd.concat(frames, axis=1).sort_index().ffill()
    # Ensure plain string column names
    daily.columns = [str(c) for c in daily.columns]
    four_h = daily.resample('4H').ffill()
    # Coerce four_h columns to strings as well (defensive)
    four_h.columns = [str(c) for c in four_h.columns]
    base_cols = [str(c) for c in daily.columns]
    for cname in base_cols:
        if cname not in four_h.columns:
            continue
        four_h[f"{cname}_pct1d"] = four_h[cname].pct_change(6)
        roll = four_h[cname].rolling(20)
        four_h[f"{cname}_z20"] = (four_h[cname] - roll.mean()) / (roll.std() + 1e-8)
    four_h = four_h.replace([np.inf, -np.inf], np.nan).ffill()
    try:
        safe_mkdir(EXTERNAL_DIR)
        four_h.to_csv(EXTERNAL_DIR / 'macro_4h.csv')
    except Exception:
        pass
    return four_h


BINANCE_PERP_MAP: Dict[str, str] = {
    'XBTUSD': 'BTCUSDT',
    'ETHUSD': 'ETHUSDT',
    'SOLUSD': 'SOLUSDT',
    'ADAUSD': 'ADAUSDT',
    'LTCUSD': 'LTCUSDT',
    'XRPUSD': 'XRPUSDT',
    'AVAXUSD': 'AVAXUSDT',
    'LINKUSD': 'LINKUSDT',
    'DOTUSD': 'DOTUSDT',
    'ATOMUSD': 'ATOMUSDT',
    'SUIUSD': 'SUIUSDT',
    'UNIUSD': 'UNIUSDT',
}


def fetch_binance_funding_4h(binance_symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch Binance perpetual funding rates (8H points), ffill to 4H, add simple stats."""
    try:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        url = 'https://fapi.binance.com/fapi/v1/fundingRate'
        out: List[dict] = []
        cursor = start_ms
        while True:
            resp = requests.get(url, params={'symbol': binance_symbol, 'startTime': cursor, 'limit': 1000}, timeout=15)
            if resp.status_code != 200:
                break
            rows = resp.json()
            if not rows:
                break
            out.extend(rows)
            last_time = rows[-1]['fundingTime']
            if last_time == cursor:
                break
            cursor = last_time + 1
            if cursor >= end_ms:
                break
            time.sleep(0.2)
        if not out:
            return pd.DataFrame()
        df = pd.DataFrame(out)
        df['time'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df.set_index('time', inplace=True)
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df['rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        four_h = df[['rate']].resample('4H').ffill()
        four_h['rate_mean_7'] = four_h['rate'].rolling(7).mean()
        four_h['rate_abs_mean_7'] = four_h['rate'].abs().rolling(7).mean()
        roll = four_h['rate'].rolling(20)
        four_h['rate_z20'] = (four_h['rate'] - roll.mean()) / (roll.std() + 1e-8)
        return four_h
    except Exception as e:
        print(f"Funding fetch failed for {binance_symbol}: {e}")
        return pd.DataFrame()


def fetch_all_funding_4h(assets: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    # Prefer OKX public API (no auth, US-friendly); fallback to Binance if needed
    for asset in assets:
        df = pd.DataFrame()
        okx_inst = OKX_PERP_MAP.get(asset)
        if okx_inst:
            df = fetch_okx_funding_4h(okx_inst)
        if (df is None or df.empty):
            b = BINANCE_PERP_MAP.get(asset)
            if b:
                df = fetch_binance_funding_4h(b)
        if df is not None and not df.empty:
            try:
                safe_mkdir(EXTERNAL_DIR)
                df.to_csv(EXTERNAL_DIR / f'funding_{asset}_4h.csv')
            except Exception:
                pass
            out[asset] = df
    return out


# OKX funding (public, no auth). Instruments like BTC-USDT-SWAP
OKX_PERP_MAP: Dict[str, str] = {
    'XBTUSD': 'BTC-USDT-SWAP',
    'ETHUSD': 'ETH-USDT-SWAP',
    'SOLUSD': 'SOL-USDT-SWAP',
    'ADAUSD': 'ADA-USDT-SWAP',
    'LTCUSD': 'LTC-USDT-SWAP',
    'XRPUSD': 'XRP-USDT-SWAP',
    'AVAXUSD': 'AVAX-USDT-SWAP',
    'LINKUSD': 'LINK-USDT-SWAP',
    'DOTUSD': 'DOT-USDT-SWAP',
    'ATOMUSD': 'ATOM-USDT-SWAP',
    'SUIUSD': 'SUI-USDT-SWAP',
    'UNIUSD': 'UNI-USDT-SWAP',
}


def fetch_okx_funding_4h(inst_id: str, days: int = 730) -> pd.DataFrame:
    """Fetch OKX funding history and resample to 4H. Pagination via 'before' cursor back in time."""
    try:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        url = 'https://www.okx.com/api/v5/public/funding-rate-history'
        all_rows: List[dict] = []
        before = None
        for _ in range(200):  # safety cap
            params = {'instId': inst_id, 'limit': '100'}
            if before:
                params['before'] = str(before)
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            payload = resp.json()
            data = payload.get('data', []) if isinstance(payload, dict) else []
            if not data:
                break
            # Determine time key present in OKX response
            def get_ts(row: dict) -> int:
                for key in ('ts', 'fundingTime', 'time'):
                    if key in row and row[key] is not None:
                        try:
                            return int(row[key])
                        except Exception:
                            pass
                return 0
            data_sorted = sorted(data, key=lambda x: get_ts(x))
            all_rows.extend(data_sorted)
            oldest_ts = get_ts(data_sorted[0]) if data_sorted else None
            if not oldest_ts or oldest_ts <= start_ms:
                break
            before = oldest_ts
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        # Build time column generically
        time_col = None
        for c in ('ts', 'fundingTime', 'time'):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            return pd.DataFrame()
        df['time'] = pd.to_datetime(pd.to_numeric(df[time_col], errors='coerce'), unit='ms')
        df.set_index('time', inplace=True)
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        # fundingRate comes as string
        rate_col = 'fundingRate' if 'fundingRate' in df.columns else None
        if rate_col is None:
            return pd.DataFrame()
        df['rate'] = pd.to_numeric(df[rate_col], errors='coerce')
        # Deduplicate timestamps before resampling
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        four_h = df[['rate']].resample('4H').ffill()
        four_h['rate_mean_7'] = four_h['rate'].rolling(7).mean()
        four_h['rate_abs_mean_7'] = four_h['rate'].abs().rolling(7).mean()
        roll = four_h['rate'].rolling(20)
        four_h['rate_z20'] = (four_h['rate'] - roll.mean()) / (roll.std() + 1e-8)
        return four_h
    except Exception as e:
        print(f"OKX funding fetch failed for {inst_id}: {e}")
        return pd.DataFrame()


def fetch_fear_greed_4h() -> pd.DataFrame:
    """Fear & Greed Index from alternative.me; daily -> 4H ffill."""
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=0&format=json', timeout=15)
        data = r.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('time', inplace=True)
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df['fng'] = pd.to_numeric(df['value'], errors='coerce')
        daily = df[['fng']].sort_index()
        four_h = daily.resample('4H').ffill()
        roll = four_h['fng'].rolling(20)
        four_h['fng_z20'] = (four_h['fng'] - roll.mean()) / (roll.std() + 1e-8)
        try:
            safe_mkdir(EXTERNAL_DIR)
            four_h.to_csv(EXTERNAL_DIR / 'fng_4h.csv')
        except Exception:
            pass
        return four_h
    except Exception as e:
        print(f"FNG fetch failed: {e}")
        return pd.DataFrame()


def fetch_mvrvz_glassnode_4h(asset_ccy: str) -> pd.DataFrame:
    """Optional MVRV Z via Glassnode (daily -> 4H). Requires GLASSNODE_API_KEY env."""
    api_key = os.environ.get('GLASSNODE_API_KEY', '')
    if not api_key:
        return pd.DataFrame()
    try:
        url = 'https://api.glassnode.com/v1/metrics/market/mvrv_z_score'
        params = {'api_key': api_key, 'a': asset_ccy, 'i': '1d'}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            print(f"Glassnode {asset_ccy} error {r.status_code}: {str(r.text)[:96]}")
            return pd.DataFrame()
        arr = r.json()
        df = pd.DataFrame(arr)
        if df.empty:
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df['t'], unit='s')
        df.set_index('time', inplace=True)
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df = df.rename(columns={'v': 'mvrvz'})[['mvrvz']]
        four_h = df.resample('4H').ffill()
        return four_h
    except Exception as e:
        print(f"MVRVZ fetch failed for {asset_ccy}: {e}")
        return pd.DataFrame()


def load_asset_1m_df(asset: str, start_date: str = '2020-01-01', append_ccxt: bool = True) -> Optional[pd.DataFrame]:
    """
    Load 1m OHLCV for asset from local Kraken CSV if available. Optionally append recent data via CCXT.
    Fallback to yfinance if CSV not found.
    Returns dataframe with columns: open, high, low, close, volume
    """
    csv_path = find_csv_for_asset(asset)
    df: Optional[pd.DataFrame] = None

    if csv_path and csv_path.exists():
        try:
            raw = pd.read_csv(str(csv_path), skiprows=1,
                              names=['time', 'open', 'high', 'low', 'close', 'volume', 'trades'],
                              low_memory=False)
            df = robust_to_datetime_index(raw, 'time', unit='s')
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna()
            print(f"Loaded CSV for {asset}: rows={len(df)}")
        except Exception as e:
            print(f"CSV load failed for {asset}: {e}")
            df = None

    if df is None:
        # yfinance fallback
        yf_ticker = ASSET_TO_YF.get(asset)
        if not yf_ticker:
            print(f"No yfinance mapping for {asset}, skipping.")
            return None
        try:
            tmp = yf.download(yf_ticker, interval='1m', period='7d', auto_adjust=False,
                              progress=False, prepost=False, threads=False, group_by='column')
            if tmp is None or tmp.empty:
                tmp = yf.download(yf_ticker, interval='1h', period='730d', auto_adjust=False,
                                  progress=False, prepost=False, threads=False, group_by='column')
                if tmp is not None and not tmp.empty:
                    tmp = tmp.resample('1min').ffill()
            if tmp is None or tmp.empty:
                tmp = yf.download(yf_ticker, interval='1d', period='5y', auto_adjust=False,
                                  progress=False, prepost=False, threads=False, group_by='column')
                if tmp is not None and not tmp.empty:
                    tmp = tmp.resample('1min').ffill()
            if tmp is None or tmp.empty:
                print(f"yfinance empty for {asset}")
                return None
            tmp.index = pd.to_datetime(tmp.index)
            try:
                tmp.index = tmp.index.tz_localize(None)
            except Exception:
                pass
            # Flatten columns if MultiIndex
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = ['_'.join([str(x) for x in c if x]) for c in tmp.columns]
            colmap = {}
            for need in ['Open', 'High', 'Low', 'Close', 'Volume']:
                match = [c for c in tmp.columns if str(c).lower() == need.lower()]
                if match:
                    colmap[need] = match[0]
            if len(colmap) < 5:
                print(f"yfinance missing columns for {asset}: have {list(tmp.columns)}")
                return None
            df = tmp[[colmap['Open'], colmap['High'], colmap['Low'], colmap['Close'], colmap['Volume']]]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            print(f"Loaded yfinance fallback for {asset}: rows={len(df)}")
        except Exception as e:
            print(f"yfinance fetch failed for {asset}: {e}")
            return None

    df = df[df.index >= pd.to_datetime(start_date)]
    if df.empty:
        return None

    # Optional CCXT append for recency
    if append_ccxt:
        try:
            exchange = ccxt.kraken()
            exchange.load_markets()
            preferred = asset.replace('USD', '/USD')
            fallbacks = ['BTC/USD'] if asset.startswith('XBT') else []
            candidates = [preferred] + [s for s in fallbacks if s]
            symbol = None
            for s in candidates:
                if s in getattr(exchange, 'markets', {}):
                    symbol = s
                    break
            if symbol is None:
                symbol = preferred  # try as-is
            since = int(df.index.max().timestamp() * 1000) + 1
            appended: List[list] = []
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', since=since, limit=1000)
                if not ohlcv:
                    break
                appended.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(1)
            if appended:
                add = pd.DataFrame(appended, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                add = robust_to_datetime_index(add, 'time', unit='ms')
                add = add[['open', 'high', 'low', 'close', 'volume']].astype(float)
                combined = pd.concat([df, add]).sort_index()
                df = combined[~combined.index.duplicated(keep='last')]
                print(f"After CCXT append for {asset}: rows={len(df)}")
        except Exception as e:
            print(f"CCXT append failed for {asset}: {e}")

    # Normalize freq and fill small gaps
    df = df.asfreq('1min').ffill().dropna()
    print(f"Final 1m DF for {asset}: rows={len(df)} range=({df.index.min()} -> {df.index.max()})")
    return df


# ----------------------------
# ICT Indicator Functions (Preserve)
# ----------------------------
def swing_highs_lows(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    highs = df['high'].rolling(window=order * 2 + 1, center=True).max() == df['high']
    lows = df['low'].rolling(window=order * 2 + 1, center=True).min() == df['low']
    df['swing_high'] = np.where(highs, df['high'], np.nan)
    df['swing_low'] = np.where(lows, df['low'], np.nan)
    return df


def fvg(df: pd.DataFrame) -> pd.DataFrame:
    df['fvg_bull'] = (df['low'] > df['high'].shift(2))
    df['fvg_bear'] = (df['high'] < df['low'].shift(2))
    df['fvg_top'] = np.where(df['fvg_bull'], df['low'], np.where(df['fvg_bear'], df['low'].shift(2), np.nan))
    df['fvg_bottom'] = np.where(df['fvg_bull'], df['high'].shift(2), np.where(df['fvg_bear'], df['high'], np.nan))
    return df


def ob(df: pd.DataFrame) -> pd.DataFrame:
    tight = (df['high'] - df['low']) < df['close'].rolling(5).std()
    df['ob_bull'] = (df['swing_low'].notna()) & tight
    df['ob_bear'] = (df['swing_high'].notna()) & tight
    df['ob_top'] = np.where(df['ob_bull'] | df['ob_bear'], df['high'], np.nan)
    df['ob_bottom'] = np.where(df['ob_bull'] | df['ob_bear'], df['low'], np.nan)
    return df


def bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    df['bos_bull'] = (df['high'] > df['swing_high'].shift(1).ffill())
    df['bos_bear'] = (df['low'] < df['swing_low'].shift(1).ffill())
    df['choch_bull'] = df['bos_bear'].shift(1) & (df['close'] > df['open'])
    df['choch_bear'] = df['bos_bull'].shift(1) & (df['close'] < df['open'])
    return df


# ----------------------------
# Vision: DINOv2 setup
# ----------------------------
def load_dino() -> Tuple[AutoImageProcessor, Dinov2Model, torch.device]:
    try:
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = Dinov2Model.from_pretrained('facebook/dinov2-base')
    except Exception as e:
        print(f"DINOv2 load warning: {e}; retrying with cache bypass")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=False)
        model = Dinov2Model.from_pretrained('facebook/dinov2-base', local_files_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    return processor, model, device


def precompute_dino_embeddings(image_paths: List[Path], out_dir: Path,
                               processor: AutoImageProcessor, model: Dinov2Model, device: torch.device) -> None:
    safe_mkdir(out_dir)
    for idx, ip in enumerate(image_paths):
        try:
            out_path = out_dir / (ip.stem + '.pt')
            if out_path.exists():
                continue
            img = Image.open(str(ip)).convert('RGB').resize((448, 448))
            inputs = processor(images=img, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            feats = outputs.last_hidden_state.mean(dim=1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1).squeeze(0).float().cpu()
            torch.save(feats, str(out_path))
            if (idx + 1) % 200 == 0:
                print(f"DINO embeddings: {idx + 1}/{len(image_paths)}")
        except Exception as e:
            print(f"DINO embedding failed for {ip}: {e}")


# ----------------------------
# YOLOv8 detection: optional inference and pseudo-labels
# ----------------------------
def maybe_load_yolo_detector():
    if YOLO_DET_MODEL_PATH and Path(YOLO_DET_MODEL_PATH).exists():
        try:
            from ultralytics import YOLO
            return YOLO(YOLO_DET_MODEL_PATH)
        except Exception as e:
            print(f"YOLO detector load failed: {e}")
    return None


def write_yolo_classes_file(root: Path) -> None:
    safe_mkdir(root)
    (root / 'classes.txt').write_text('\n'.join(YOLO_CLASSES))


def ict_to_yolo_boxes(window: pd.DataFrame, seq_len: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Create YOLO-format boxes (class_id, x_center, y_center, width, height), all normalized to [0,1],
    from ICT signals inside the sliding window. The x/y normalization uses candle index and price range.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    if window.empty:
        return boxes
    price_min = float(window['low'].min())
    price_max = float(window['high'].max())
    price_range = max(price_max - price_min, 1e-8)

    def norm_x(i: int) -> float:
        return (i + 0.5) / seq_len

    def norm_y(p: float) -> float:
        return (p - price_min) / price_range

    # FVG boxes
    for i, row in enumerate(window.itertuples(index=False)):
        if getattr(row, 'fvg_bull'):
            top = getattr(row, 'fvg_top')
            bottom = getattr(row, 'fvg_bottom')
            if not (np.isnan(top) or np.isnan(bottom)):
                y1, y2 = sorted([float(top), float(bottom)])
                yc = norm_y((y1 + y2) / 2.0)
                h = max(norm_y(y2) - norm_y(y1), 0.01)
                boxes.append((YOLO_CLASSES.index('fvg_bull'), norm_x(i), yc, 1.2 / seq_len, h))
        if getattr(row, 'fvg_bear'):
            top = getattr(row, 'fvg_top')
            bottom = getattr(row, 'fvg_bottom')
            if not (np.isnan(top) or np.isnan(bottom)):
                y1, y2 = sorted([float(top), float(bottom)])
                yc = norm_y((y1 + y2) / 2.0)
                h = max(norm_y(y2) - norm_y(y1), 0.01)
                boxes.append((YOLO_CLASSES.index('fvg_bear'), norm_x(i), yc, 1.2 / seq_len, h))

    # OB boxes (use ob_top/bottom levels)
    for i, row in enumerate(window.itertuples(index=False)):
        if getattr(row, 'ob_bull') or getattr(row, 'ob_bear'):
            top = getattr(row, 'ob_top')
            bottom = getattr(row, 'ob_bottom')
            if not (np.isnan(top) or np.isnan(bottom)):
                y1, y2 = sorted([float(top), float(bottom)])
                yc = norm_y((y1 + y2) / 2.0)
                h = max(norm_y(y2) - norm_y(y1), 0.01)
                cls_name = 'ob_bull' if getattr(row, 'ob_bull') else 'ob_bear'
                boxes.append((YOLO_CLASSES.index(cls_name), norm_x(i), yc, 1.2 / seq_len, h))

    # BOS/CHoCH boxes (small markers near candle extremes)
    for i, row in enumerate(window.itertuples(index=False)):
        if getattr(row, 'bos_bull'):
            y = float(getattr(row, 'high'))
            boxes.append((YOLO_CLASSES.index('bos_bull'), norm_x(i), norm_y(y), 1.0 / seq_len, 0.02))
        if getattr(row, 'bos_bear'):
            y = float(getattr(row, 'low'))
            boxes.append((YOLO_CLASSES.index('bos_bear'), norm_x(i), norm_y(y), 1.0 / seq_len, 0.02))
        if getattr(row, 'choch_bull'):
            y = float(getattr(row, 'close'))
            boxes.append((YOLO_CLASSES.index('choch_bull'), norm_x(i), norm_y(y), 1.0 / seq_len, 0.02))
        if getattr(row, 'choch_bear'):
            y = float(getattr(row, 'close'))
            boxes.append((YOLO_CLASSES.index('choch_bear'), norm_x(i), norm_y(y), 1.0 / seq_len, 0.02))

    # Clip values to [0,1]
    clipped: List[Tuple[int, float, float, float, float]] = []
    for cls_id, xc, yc, w, h in boxes:
        xc = float(np.clip(xc, 0.0, 1.0))
        yc = float(np.clip(yc, 0.0, 1.0))
        w = float(np.clip(w, 0.005, 1.0))
        h = float(np.clip(h, 0.005, 1.0))
        clipped.append((cls_id, xc, yc, w, h))
    return clipped


def write_yolo_label_file(labels_dir: Path, image_stem: str, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    safe_mkdir(labels_dir)
    out_path = labels_dir / f"{image_stem}.txt"
    with open(out_path, 'w') as f:
        for cls_id, xc, yc, w, h in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ----------------------------
# Chart generation and metadata
# ----------------------------
def summarize_change(window_end_close: float, future_close_short: float, future_close_long: float) -> float:
    change_short = (future_close_short - window_end_close) / (window_end_close + 1e-8)
    change_long = (future_close_long - window_end_close) / (window_end_close + 1e-8)
    return float((change_short + change_long) / 2.0)


def compute_quantile_thresholds(changes: List[float]) -> Tuple[float, float]:
    if not changes:
        return 0.33, 0.67
    arr = np.array(changes, dtype=np.float32)
    bear = float(np.quantile(arr, 0.33))
    bull = float(np.quantile(arr, 0.67))
    # Widen thresholds if distribution is too tight
    if abs(bull - bear) < 1e-6:
        bear = float(np.quantile(arr, 0.40))
        bull = float(np.quantile(arr, 0.60))
    return bear, bull


def plot_window_with_overlays(window: pd.DataFrame, figsize=FIGSIZE, dpi=DPI):
    addplots = [mpf.make_addplot(window['close'].rolling(20).mean(), color='blue', width=1)]
    fvg_bull_plot = pd.Series(np.nan, index=window.index)
    fvg_bull_plot[window['fvg_bull']] = window.loc[window['fvg_bull'], 'fvg_top']
    if window['fvg_bull'].any():
        addplots.append(mpf.make_addplot(fvg_bull_plot, type='scatter', markersize=50, marker='^', color='green'))
    fvg_bear_plot = pd.Series(np.nan, index=window.index)
    fvg_bear_plot[window['fvg_bear']] = window.loc[window['fvg_bear'], 'fvg_bottom']
    if window['fvg_bear'].any():
        addplots.append(mpf.make_addplot(fvg_bear_plot, type='scatter', markersize=50, marker='v', color='red'))
    ob_levels = window[(window['ob_bull'] | window['ob_bear'])]['ob_top'].dropna().unique().tolist() + \
                window[(window['ob_bull'] | window['ob_bear'])]['ob_bottom'].dropna().unique().tolist()
    hlines_dict = None
    if ob_levels:
        hlines_dict = dict(hlines=ob_levels, colors='blue', linestyle='--', alpha=0.5)
    kwargs = dict(type='candle', style='charles', returnfig=True, figsize=figsize, volume=True, addplot=addplots, axisoff=True)
    if hlines_dict:
        kwargs['hlines'] = hlines_dict
    fig, ax = mpf.plot(window, **kwargs)
    fig.canvas.draw()
    return fig


def generate_for_asset_interval(asset: str, df_1m: pd.DataFrame, minutes: int,
                                 meta_accum: Dict[str, dict]) -> None:
    interval_str = minutes_to_pandas_freq(minutes)
    # Down/Up-sample to target interval
    if minutes == 1:
        df = df_1m.copy()
    else:
        freq = interval_str
        df = pd.DataFrame({
            'open': df_1m['open'].resample(freq).first(),
            'high': df_1m['high'].resample(freq).max(),
            'low': df_1m['low'].resample(freq).min(),
            'close': df_1m['close'].resample(freq).last(),
            'volume': df_1m['volume'].resample(freq).sum(),
        }).dropna()

    if df.empty:
        print(f"Empty DF for {asset} @ {interval_str}")
        return

    # ICT indicators on full DF
    df = swing_highs_lows(df)
    df = fvg(df)
    df = ob(df)
    df = bos_choch(df)
    print(f"Resampled {asset}@{interval_str}: bars={len(df)} range=({df.index.min()} -> {df.index.max()})")

    # Window pre-pass to compute avg_change distribution (no volatility filter here)
    max_i = max(0, len(df) - SEQ_LEN - FORECAST_HORIZON_BARS)
    changes: List[float] = []
    for i in range(0, max_i, STEP):
        window = df.iloc[i:i + SEQ_LEN]
        if len(window) < SEQ_LEN:
            continue
        end_close = float(df.iloc[i + SEQ_LEN]['close'])
        f_short = float(df.iloc[min(i + SEQ_LEN + 5, len(df) - 1)]['close'])
        f_long = float(df.iloc[min(i + SEQ_LEN + 20, len(df) - 1)]['close'])
        changes.append(summarize_change(end_close, f_short, f_long))
    if not changes:
        print(f"No valid windows for {asset}@{interval_str}")
        return
    bear_thr, bull_thr = compute_quantile_thresholds(changes)

    # Ensure dirs
    for lab in ['bullish', 'bearish', 'neutral']:
        safe_mkdir(DATASET_DIR / lab)
    safe_mkdir(WINDOWS_DIR)
    safe_mkdir(WINDOWS_FWD_DIR)
    safe_mkdir(EXTERNAL_DIR)
    write_yolo_classes_file(YOLO_DET_DIR)
    safe_mkdir(YOLO_DET_DIR / 'images')
    safe_mkdir(YOLO_DET_DIR / 'labels')

    # Per-interval volatility threshold (looser for higher TFs)
    if minutes >= 1440:
        vol_std_min = 0.00005
    elif minutes >= 240:
        vol_std_min = 0.0002
    else:
        vol_std_min = 0.003

    generated = 0
    label_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    for i in range(0, max_i, STEP):
        if generated >= MAX_WINDOWS_PER_COMBO:
            break
        window = df.iloc[i:i + SEQ_LEN].copy()
        if len(window) < SEQ_LEN:
            continue
        # Filters
        if window['close'].pct_change().dropna().std() < vol_std_min:
            continue
        end_close = float(df.iloc[i + SEQ_LEN]['close'])
        f_short = float(df.iloc[min(i + SEQ_LEN + 5, len(df) - 1)]['close'])
        f_long = float(df.iloc[min(i + SEQ_LEN + 20, len(df) - 1)]['close'])
        avg_change = summarize_change(end_close, f_short, f_long)

        has_bull = int(window['fvg_bull'].any()) + int(window['ob_bull'].any()) + int(window['bos_bull'].any())
        has_bear = int(window['fvg_bear'].any()) + int(window['ob_bear'].any()) + int(window['bos_bear'].any())
        ict_conf = max(has_bull, has_bear)

        label: Optional[str] = None
        # Label purely by price-action quantiles; ICT signals are recorded but not required
        if avg_change > bull_thr:
            label = 'bullish'
        elif avg_change < bear_thr:
            label = 'bearish'
        else:
            label = 'neutral'
        if label is None:
            continue

        # Plot figure and save
        fig = plot_window_with_overlays(window)
        img_name = f"{asset}_{minutes}m_{window.index[0].strftime('%Y%m%d%H%M%S')}.png"
        img_path = DATASET_DIR / label / img_name
        buf_path = DATASET_DIR / '._tmp.png'
        try:
            fig.savefig(str(buf_path), format='png', dpi=DPI, bbox_inches='tight', pad_inches=0)
        finally:
            plt.close(fig)
            plt.close('all')
        img = Image.open(str(buf_path)).convert('RGB')
        img.save(str(img_path))
        # simple augmentations
        ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1)).save(str(DATASET_DIR / label / (img_path.stem + '_b.png')))
        ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1)).save(str(DATASET_DIR / label / (img_path.stem + '_c.png')))
        try:
            buf_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

        # Save window pickle
        window_fn = f"{asset}_{minutes}m_{window.index[0].strftime('%Y%m%d%H%M%S')}.pkl"
        window_path = WINDOWS_DIR / window_fn
        try:
            window.to_pickle(str(window_path))
        except Exception as e:
            print(f"Failed to save window {window_fn}: {e}")

        # Save forward slice for exit simulation (time/trailed stops)
        fwd_start = i + SEQ_LEN
        fwd_end = min(i + SEQ_LEN + FORECAST_HORIZON_BARS, len(df))
        future_df = df.iloc[fwd_start:fwd_end].copy()
        future_fn = f"{asset}_{minutes}m_{window.index[0].strftime('%Y%m%d%H%M%S')}_fwd.pkl"
        future_path = WINDOWS_FWD_DIR / future_fn
        try:
            future_df.to_pickle(str(future_path))
        except Exception as e:
            print(f"Failed to save future {future_fn}: {e}")

        # YOLO pseudo-labels from ICT
        boxes = ict_to_yolo_boxes(window, SEQ_LEN)
        # copy image into YOLO images dir and write label
        yolo_img_dst = YOLO_DET_DIR / 'images' / img_path.name
        shutil.copyfile(str(img_path), str(yolo_img_dst))
        write_yolo_label_file(YOLO_DET_DIR / 'labels', img_path.stem, boxes)

        # Optional: run YOLO detector on the image (if model provided) and record detections in metadata
        yolo_dets: List[dict] = []
        try:
            _yolo = maybe_load_yolo_detector()
            if _yolo is not None:
                res = _yolo(str(img_path))
                try:
                    boxes_xywhn = res[0].boxes.xywhn.cpu().numpy()
                    cls_ids = res[0].boxes.cls.cpu().numpy()
                    confs = res[0].boxes.conf.cpu().numpy()
                    for (xc, yc, w, h), ci, cf in zip(boxes_xywhn, cls_ids, confs):
                        yolo_dets.append({
                            'cls_id': int(ci), 'conf': float(cf), 'xc': float(xc), 'yc': float(yc), 'w': float(w), 'h': float(h)
                        })
                except Exception:
                    pass
        except Exception as e:
            print(f"YOLO inference failed on {img_path.name}: {e}")

        # Metadata
        meta_accum[img_path.name] = {
            'timestamp': window.index[0].isoformat(),
            'asset': asset,
            'interval_min': minutes,
            'label': label,
            'avg_change': avg_change,
            'window_file': window_fn,
            'future_file': future_fn,
            'ict_counts': {
                'fvg_bull': int(window['fvg_bull'].sum()),
                'fvg_bear': int(window['fvg_bear'].sum()),
                'ob_bull': int(window['ob_bull'].sum()),
                'ob_bear': int(window['ob_bear'].sum()),
                'bos_bull': int(window['bos_bull'].sum()),
                'bos_bear': int(window['bos_bear'].sum()),
                'choch_bull': int(window['choch_bull'].sum()),
                'choch_bear': int(window['choch_bear'].sum()),
            },
            'yolo_pseudo_boxes': boxes,  # ICT-derived
            'yolo_detections': yolo_dets,  # model-derived if available
        }

        generated += 1
        label_counts[label] += 1
    print(f"Generated {generated} windows for {asset}@{interval_str} | counts: {label_counts}")


# ----------------------------
# V2 additional helpers and main
# ----------------------------
try:
    from features.microstructure import add_microstructure  # type: ignore
except Exception:
    def add_microstructure(bar: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bar.index)
        try:
            ret = bar['close'].pct_change()
            out['ret'] = ret
            out['vol'] = bar['volume'].fillna(0.0)
            out['range'] = (bar['high'] - bar['low']).fillna(0.0)
        except Exception:
            pass
        return out
try:
    from features.volatility import add_har_rv  # type: ignore
except Exception:
    def add_har_rv(returns: pd.Series) -> pd.DataFrame:
        r2 = (returns.fillna(0.0))**2
        out = pd.DataFrame(index=returns.index)
        out['rv_d'] = r2.rolling(24).sum()
        out['rv_w'] = r2.rolling(24*7).sum()
        out['rv_m'] = r2.rolling(24*30).sum()
        return out
try:
    from features.regime import add_regime  # type: ignore
except Exception:
    def add_regime(close: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=close.index)
        ma_fast = close.rolling(20).mean(); ma_slow = close.rolling(50).mean()
        out['trend'] = (ma_fast - ma_slow) / (ma_slow + 1e-8)
        return out
try:
    from features.funding_basis import add_funding_basis  # type: ignore
except Exception:
    def add_funding_basis(bar: pd.DataFrame, funding=None, basis=None) -> pd.DataFrame:
        return pd.DataFrame(index=bar.index)

def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    return tr.rolling(14).mean()

def resample_to_interval(df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {'1m':'1T','5m':'5T','15m':'15T','1h':'1H','4h':'4H','1d':'1D','4H':'4H','1D':'1D'}.get(interval, '1H')
    df = pd.DataFrame({
        'open': df_1m['open'].resample(rule).first(),
        'high': df_1m['high'].resample(rule).max(),
        'low': df_1m['low'].resample(rule).min(),
        'close': df_1m['close'].resample(rule).last(),
        'volume': df_1m['volume'].resample(rule).sum(),
    }).dropna()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    # full reindex to uniform grid
    idx = pd.date_range(df.index.min(), df.index.max(), freq=rule)
    df = df.reindex(idx)
    for c in ['open','high','low','close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0.0)
    return df.dropna()

def build_per_bar_features(asset: str, bar: pd.DataFrame, enable_flags: dict, rebuild: bool) -> pd.DataFrame:
    fpath = DATASET_DIR / f"feats_bar_{asset}.parquet"
    if fpath.exists() and not rebuild:
        try:
            return pd.read_parquet(fpath)
        except Exception:
            pass
    feats = []
    if enable_flags.get('microstructure', True): feats.append(add_microstructure(bar))
    if enable_flags.get('volatility_har', True): feats.append(add_har_rv(bar['close'].pct_change()))
    if enable_flags.get('regime', True): feats.append(add_regime(bar['close']))
    if enable_flags.get('funding_basis', True): feats.append(add_funding_basis(bar, funding=None, basis=None))
    feat_bar = pd.concat(feats, axis=1) if len(feats) else pd.DataFrame(index=bar.index)
    out = pd.concat([bar[['open','high','low','close','volume']], feat_bar], axis=1)
    out = out.reindex(bar.index)
    out.to_parquet(fpath)
    return out

def window_stats(df_feat: pd.DataFrame, start: int, end: int) -> Dict[str, float]:
    w = df_feat.iloc[start:end]
    agg: Dict[str, float] = {}
    if w.empty:
        return agg
    for c in w.columns:
        s = w[c].astype(float)
        agg[f"{c}_mean"] = float(s.mean()); agg[f"{c}_std"] = float(s.std()); agg[f"{c}_last"] = float(s.iloc[-1])
        try:
            y = s.values; x = np.arange(len(y)); b = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
            agg[f"{c}_slope"] = float(b)
        except Exception:
            agg[f"{c}_slope"] = 0.0
    return agg

def relax_quantiles_for_balance(y: np.ndarray, q_low: float, q_high: float, min_frac: float = 0.05) -> Tuple[float, float]:
    for d in [0.0, 0.02, 0.05, 0.08, 0.10]:
        lo = max(0.05, q_low - d); hi = min(0.95, q_high + d)
        lo_thr = float(np.quantile(y, lo)); hi_thr = float(np.quantile(y, hi))
        cls = np.where(y > hi_thr, 1, np.where(y < lo_thr, -1, 0))
        n = len(y); 
        if n == 0: return q_low, q_high
        if min((cls == -1).mean(), (cls == 0).mean(), (cls == 1).mean()) >= min_frac:
            return lo, hi
    return q_low, q_high

def build_windows_and_labels(asset: str, bar: pd.DataFrame, feat_bar: pd.DataFrame, lookback: int, horizon: int, dino_images: bool, ict_counts: bool, quick: bool) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    close = bar['close'].astype(float)
    fut = (close.shift(-horizon) / close - 1.0).astype(np.float32)
    atr = compute_atr14(bar).astype(np.float32)
    rows = []; legacy_meta: Dict[str, dict] = {}
    idx = bar.index; n = len(bar)
    split = int(0.8 * n); y_train = fut.iloc[:split].dropna().values
    ql, qh = (0.33, 0.67)
    if len(y_train) > 0:
        ql, qh = relax_quantiles_for_balance(y_train, ql, qh, 0.05)
    for end_idx in range(lookback, n - horizon):
        start_idx = end_idx - lookback; t = idx[end_idx]
        ret_fut = float(fut.iloc[end_idx]) if not np.isnan(fut.iloc[end_idx]) else 0.0
        lab_dir = 1 if ret_fut > 0 else 0
        if ret_fut > np.quantile(y_train, qh) if len(y_train) else ret_fut > 0:
            lab_mv = 'bullish'
        elif ret_fut < np.quantile(y_train, ql) if len(y_train) else ret_fut < 0:
            lab_mv = 'bearish'
        else:
            lab_mv = 'neutral'
        agg = window_stats(feat_bar, start_idx, end_idx)
        agg.update({'t': t, 'ret_fut': ret_fut, 'y_direction': int(lab_dir), 'y_movement': lab_mv, 'move_abs': abs(ret_fut), 'atr_now': float(atr.iloc[end_idx]) if not np.isnan(atr.iloc[end_idx]) else 0.0})
        rows.append(agg)
        try:
            WINDOWS_DIR.mkdir(parents=True, exist_ok=True); WINDOWS_FWD_DIR.mkdir(parents=True, exist_ok=True)
            window = bar.iloc[start_idx:end_idx].copy(); future = bar.iloc[end_idx:end_idx+horizon].copy()
            ts_str = pd.Timestamp(t).strftime('%Y%m%d%H%M%S')
            window_fn = f"{asset}_{CONF['data']['interval']}_{ts_str}.pkl"; future_fn = f"{asset}_{CONF['data']['interval']}_{ts_str}_fwd.pkl"
            window.to_pickle(str(WINDOWS_DIR / window_fn)); future.to_pickle(str(WINDOWS_FWD_DIR / future_fn))
            img_name = f"{asset}_{CONF['data']['interval']}_{ts_str}.png"
            if dino_images or ict_counts:
                try:
                    fig = mpf.plot(window, type='candle', style='charles', returnfig=True, figsize=(6,4), volume=False, axisoff=True)[0]
                    buf_path = DATASET_DIR / '._tmp_part0v2.png'
                    fig.savefig(str(buf_path), format='png', dpi=200, bbox_inches='tight', pad_inches=0)
                    plt.close(fig); plt.close('all')
                    (DATASET_DIR / lab_mv).mkdir(parents=True, exist_ok=True)
                    Image.open(str(buf_path)).convert('RGB').save(str((DATASET_DIR / lab_mv / img_name)))
                    try: buf_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        import logging
                        logging.exception("Cleanup temp image failed in Part0")
                except Exception:
                    import logging
                    logging.exception("Figure save or image processing failed in Part0 dino_images/ict_counts block")
            legacy_meta[img_name] = {'timestamp': pd.Timestamp(t).isoformat(), 'asset': asset, 'interval_min': 60 if CONF['data']['interval'] in ('1h','1H') else (240 if CONF['data']['interval'] in ('4h','4H') else 1440 if CONF['data']['interval'] in ('1d','1D') else 1), 'label': lab_mv, 'avg_change': ret_fut, 'window_file': window_fn, 'future_file': future_fn, 'yolo_pseudo_boxes': [], 'external_features': {}}
        except Exception:
            import logging
            logging.exception("Window/future save or metadata population failed in Part0 loop")
    win_df = pd.DataFrame(rows)
    if not win_df.empty:
        win_df['t'] = pd.to_datetime(win_df['t']); win_df = win_df.sort_values('t')
    return win_df, legacy_meta

def triple_barrier_events(asset: str, bar: pd.DataFrame, horizon: int, ub_mult: float = 2.5, lb_mult: float = 0.75) -> pd.DataFrame:
    atr = compute_atr14(bar); idx = bar.index; out = []
    for i in range(len(bar) - horizon):
        entry = float(bar['close'].iloc[i]); atr_now = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        ub = entry + ub_mult * atr_now; lb = entry - lb_mult * atr_now
        path = bar['close'].iloc[i:i+horizon+1].astype(float)
        hit = 'none'; exit_t = idx[min(i+horizon, len(bar)-1)]
        if (path >= ub).any(): hit = 'upper'; exit_t = path[path >= ub].index[0]
        if (path <= lb).any():
            t_hit = path[path <= lb].index[0]
            if hit == 'none' or t_hit < exit_t: hit = 'lower'; exit_t = t_hit
        out.append({'t': idx[i], 'entry': entry, 'atr': atr_now, 'ub': ub, 'lb': lb, 'exit_t': exit_t, 'label': hit})
    return pd.DataFrame(out)

def make_splits(win_df: pd.DataFrame, folds: int, embargo_bars: int) -> List[Dict[str, List[int]]]:
    if win_df.empty: return []
    n = len(win_df); fold_sizes = [n // folds] * folds
    for i in range(n % folds): fold_sizes[i] += 1
    splits = []; start = 0
    for fs in fold_sizes:
        val_idx = np.arange(start, start+fs)
        tr_left = np.arange(0, max(0, start - embargo_bars))
        tr_right = np.arange(min(n, start + fs + embargo_bars), n)
        splits.append({'train': np.concatenate([tr_left, tr_right]).tolist(), 'val': val_idx.tolist()})
        start += fs
    return splits

def main():
    warnings.filterwarnings('ignore')
    safe_mkdir(DATASET_DIR); safe_mkdir(WINDOWS_DIR); safe_mkdir(WINDOWS_FWD_DIR)
    import argparse as _arg
    ap = _arg.ArgumentParser()
    ap.add_argument("--config", default="conf/experiment.yaml"); ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    global CONF
    if args.config and args.config != "conf/experiment.yaml": CONF = load_config(args.config)
    if args.symbols: CONF['data']['symbols'] = args.symbols
    if args.start: CONF['data']['start'] = args.start
    if args.end: CONF['data']['end'] = args.end
    if args.quick:
        CONF['training']['folds'] = 1
        CONF['features']['dino_embeddings'] = False
        CONF['features']['ict_counts'] = False

    symbols = CONF['data'].get('symbols', [])
    interval = CONF['data'].get('interval', '1h')
    lookback = int(CONF['data'].get('lookback_bars', 240))
    horizon = int(CONF['data'].get('horizon_bars', 12))
    folds = int(CONF['training'].get('folds', 5))
    embargo = int(CONF['training'].get('embargo_bars', 10))
    enable_flags = CONF.get('features', {})

    legacy_meta_agg: Dict[str, dict] = {}
    per_asset_counts: Dict[str, int] = {}
    total_samples = 0
    images_enabled = bool(enable_flags.get('dino_embeddings', False) or enable_flags.get('ict_counts', False))

    for asset in symbols:
        print(f"[Part0] Loading data for {asset} ...")
        df_1m = load_asset_1m_df(asset, start_date=CONF['data'].get('start', '2020-01-01'), append_ccxt=True)
        if df_1m is None or df_1m.empty:
            print(f"No data for {asset}, skipping."); continue
        bar = resample_to_interval(df_1m, interval)
        if args.quick: bar = bar.iloc[: min(len(bar), 10000)]
        feat_bar = build_per_bar_features(asset, bar, enable_flags, rebuild=args.rebuild)
        win_df, legacy_meta = build_windows_and_labels(asset, bar, feat_bar, lookback, horizon, images_enabled, enable_flags.get('ict_counts', False), args.quick)
        if not win_df.empty:
            win_df.to_parquet(DATASET_DIR / f"windows_{asset}.parquet")
        ev = triple_barrier_events(asset, bar, horizon)
        if not ev.empty:
            ev.to_parquet(DATASET_DIR / f"events_{asset}.parquet")
        per_asset_counts[asset] = int(len(win_df)); total_samples += int(len(win_df)); legacy_meta_agg.update(legacy_meta)

    if total_samples == 0:
        raise RuntimeError("No samples; check date range, quantiles, or feature availability.")
    for a, n in per_asset_counts.items():
        if n < 200 and not args.quick:
            print(f"Warning: {a} windows={n} < 200")

    # per-asset splits
    splits = {"folds": folds, "embargo_bars": embargo, "per_asset": {}}
    for a in per_asset_counts.keys():
        p = DATASET_DIR / f"windows_{a}.parquet"
        if not p.exists(): continue
        w = pd.read_parquet(p)
        if 't' in w.columns: w = w.sort_values('t')
        splits["per_asset"][a] = make_splits(w, folds, embargo)
    with open(DATASET_DIR / 'splits.json', 'w') as f:
        json.dump(splits, f)

    meta = {
        "version": "part0_v2", "created_at": datetime.utcnow().isoformat(),
        "symbols": symbols, "interval": interval, "lookback_bars": lookback, "horizon_bars": horizon,
        "n_samples_total": int(total_samples), "per_asset_counts": per_asset_counts,
        "features": {"microstructure": bool(enable_flags.get('microstructure', True)), "volatility_har": bool(enable_flags.get('volatility_har', True)), "regime": bool(enable_flags.get('regime', True)), "funding_basis": bool(enable_flags.get('funding_basis', True)), "dino_embeddings": bool(enable_flags.get('dino_embeddings', False)), "ict_counts": bool(enable_flags.get('ict_counts', False))},
        "labels": {"movement_quantiles": {"q_low": 0.33, "q_high": 0.67, "chosen_low": None, "chosen_high": None}, "class_balance": None},
        "splits": {"folds": folds, "embargo_bars": embargo},
        "paths": {"feats_bar": "feats_bar_<ASSET>.parquet", "windows": "windows_<ASSET>.parquet", "labels": "labels_<ASSET>.parquet", "images_root": "images/", "dino_embeddings": "dino_<ASSET>.npy", "splits": "splits.json"}
    }
    meta.update(legacy_meta_agg)
    with open(DATASET_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f)
    print(f"[Part0] Done. samples_total={total_samples}, assets={len(per_asset_counts)}, img_enabled={images_enabled}, dino_enabled={bool(enable_flags.get('dino_embeddings', False))}")
    print(f"Outputs: {DATASET_DIR}")


def main():
    warnings.filterwarnings('ignore')
    safe_mkdir(BASE_DIR)
    safe_mkdir(DATASET_DIR)
    safe_mkdir(WINDOWS_DIR)
    safe_mkdir(EMBED_DIR)
    safe_mkdir(YOLO_DET_DIR)

    try_extract_zip_if_needed()

    metadata: Dict[str, dict] = {}

    # Pre-fetch external panels once (4H grid), then attach per-sample
    print("Fetching external macro/derivatives context ...")
    macro_4h = fetch_macro_4h()
    fng_4h = fetch_fear_greed_4h()
    funding_map = fetch_all_funding_4h(ASSETS)
    # Optional on-chain (MVRV Z) for BTC, ETH
    mvrvz_btc = fetch_mvrvz_glassnode_4h('BTC')
    mvrvz_eth = fetch_mvrvz_glassnode_4h('ETH')
    for asset in ASSETS:
        print(f"Loading 1m data for {asset} ...")
        df_1m = load_asset_1m_df(asset, start_date='2020-01-01', append_ccxt=True)
        if df_1m is None or df_1m.empty:
            print(f"No data for {asset}, skipping.")
            continue
        for minutes in INTERVALS_MIN:
            print(f"Generating windows for {asset} @ {minutes}m ...")
            generate_for_asset_interval(asset, df_1m, minutes, metadata)

    # Save metadata
    meta_path = DATASET_DIR / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved: {meta_path}")

    # Enrich metadata entries with nearest external features at timestamp
    def external_for(asset: str, ts: datetime) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # Funding (per-asset)
        fdf = funding_map.get(asset)
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            sub = fdf.loc[:ts]
            if not sub.empty:
                for k, v in sub.iloc[-1].items():
                    out[f'fund_{k}'] = float(v) if isinstance(v, (int,float,np.floating)) and not math.isnan(v) else None
        # Macro
        if isinstance(macro_4h, pd.DataFrame) and not macro_4h.empty:
            sub = macro_4h.loc[:ts]
            if not sub.empty:
                for k, v in sub.iloc[-1].items():
                    out[f'macro_{k}'] = float(v) if isinstance(v, (int,float,np.floating)) and not math.isnan(v) else None
        # FNG
        if isinstance(fng_4h, pd.DataFrame) and not fng_4h.empty:
            sub = fng_4h.loc[:ts]
            if not sub.empty:
                for k, v in sub.iloc[-1].items():
                    out[f'fng_{k}'] = float(v) if isinstance(v, (int,float,np.floating)) and not math.isnan(v) else None
        # MVRVZ (BTC/ETH only)
        if asset == 'XBTUSD' and isinstance(mvrvz_btc, pd.DataFrame) and not mvrvz_btc.empty:
            sub = mvrvz_btc.loc[:ts]
            if not sub.empty:
                out['onchain_mvrvz'] = float(sub.iloc[-1]['mvrvz']) if not math.isnan(sub.iloc[-1]['mvrvz']) else None
        if asset == 'ETHUSD' and isinstance(mvrvz_eth, pd.DataFrame) and not mvrvz_eth.empty:
            sub = mvrvz_eth.loc[:ts]
            if not sub.empty:
                out['onchain_mvrvz'] = float(sub.iloc[-1]['mvrvz']) if not math.isnan(sub.iloc[-1]['mvrvz']) else None
        return out

    updated = 0
    with open(meta_path, 'r') as f:
        meta_all = json.load(f)
    for k, m in meta_all.items():
        try:
            asset = m.get('asset')
            ts = pd.to_datetime(m.get('timestamp')).to_pydatetime()
            m['external_features'] = external_for(asset, ts)
            updated += 1
        except Exception:
            pass
    with open(meta_path, 'w') as f:
        json.dump(meta_all, f)
    print(f"External features attached to {updated} metadata entries.")

    # Precompute DINO embeddings for all images
    print("Loading DINOv2 and precomputing embeddings...")
    processor, dino, device = load_dino()
    image_paths: List[Path] = []
    for lab in ['bullish', 'bearish', 'neutral']:
        lab_dir = DATASET_DIR / lab
        if lab_dir.exists():
            image_paths.extend(sorted(p for p in lab_dir.glob('*.png')))
    precompute_dino_embeddings(image_paths, EMBED_DIR, processor, dino, device)
    print("DINO embeddings complete.")

    # Write a simple images manifest for downstream vision fine-tuning
    try:
        manifest_rows: List[str] = []
        header = 'image_path,label,timestamp,asset,interval_min,avg_change'\
            + ',has_yolo_boxes,has_external_features'\
            + '\n'
        manifest_rows.append(header)
        for fname, m in meta_all.items():
            lab = m.get('label', '')
            ts = m.get('timestamp', '')
            asset = m.get('asset', '')
            interval_min = m.get('interval_min', '')
            avg_change = m.get('avg_change', '')
            img_rel = f"{lab}/{fname}"
            has_boxes = int(bool(m.get('yolo_pseudo_boxes')))
            has_ext = int(bool(m.get('external_features')))
            row = f"{img_rel},{lab},{ts},{asset},{interval_min},{avg_change},{has_boxes},{has_ext}\n"
            manifest_rows.append(row)
        (DATASET_DIR / 'images_manifest.csv').write_text(''.join(manifest_rows))
        print(f"Images manifest written: {DATASET_DIR / 'images_manifest.csv'}")
    except Exception as e:
        print(f"Images manifest write failed: {e}")

    # Write YOLO classes file (again to ensure present)
    write_yolo_classes_file(YOLO_DET_DIR)
    print(f"YOLO detection dataset prepared at: {YOLO_DET_DIR}")
    print("Part 0 complete.")

    # ----------------------------
    # PART 0.B: Fine-tuning Vision Models (optional)
    # ----------------------------
    try:
        # YOLOv8 fine-tuning on pseudo-labels (ICT)
        images_dir = YOLO_DET_DIR / 'images'
        labels_dir = YOLO_DET_DIR / 'labels'
        if images_dir.exists() and labels_dir.exists() and any(images_dir.glob('*.png')):
            from ultralytics import YOLO
            # Prepare Ultralytics dataset structure: a simple data.yaml in workspace
            data_yaml = BASE_DIR / 'yolo_ict_data.yaml'
            data_yaml.write_text(f"""
path: {YOLO_DET_DIR}
train: images
val: images
names: {YOLO_CLASSES}
            """.strip())
            print(f"YOLO data.yaml written to {data_yaml}")
            model = YOLO('yolov8n.pt')
            device = 0 if torch.cuda.is_available() else 'cpu'
            print(f"Starting YOLOv8 fine-tune on device={device} ...")
            model.train(data=str(data_yaml), epochs=12, imgsz=640, batch=16, device=device, workers=2)
            # Save best weights to BASE_DIR
            # Ultralytics saves runs under runs/detect/train*/weights/best.pt
            best = None
            try:
                from glob import glob
                cands = sorted(glob('runs/detect/*/weights/best.pt'), key=os.path.getmtime)
                if cands:
                    best = cands[-1]
            except Exception:
                pass
            out_w = BASE_DIR / 'finetuned_yolo_ict.pt'
            if best and os.path.isfile(best):
                shutil.copy(best, out_w)
                print(f"Saved finetuned YOLO weights to {out_w}")
            else:
                try:
                    model.model.save(str(out_w))  # fallback
                    print(f"Saved YOLO model to {out_w}")
                except Exception as e:
                    print(f"Failed to save YOLO model: {e}")
            # Set for future runs
            os.environ['YOLO_ICT_MODEL_PATH'] = str(out_w)
            print(f"Set YOLO_ICT_MODEL_PATH={out_w}")
        else:
            print("YOLO fine-tune skipped: no images/labels found.")
    except Exception as e:
        print(f"YOLO fine-tuning error: {e}")

    try:
        # DINOv2 classification fine-tuning (bull/bear/neutral)
        # Build a small dataset list from metadata
        meta_fp = DATASET_DIR / 'metadata.json'
        if meta_fp.exists():
            with open(meta_fp, 'r') as f:
                meta_all = json.load(f)
            # Build splits by timestamp (temporal split)
            entries = []
            for fname, m in meta_all.items():
                lab = m.get('label')
                if lab not in {'bullish','bearish','neutral'}:
                    continue
                # Resolve path
                img_path = DATASET_DIR / lab / fname
                if not img_path.exists():
                    continue
                ts = m.get('timestamp', '')
                entries.append((ts, str(img_path), {'bullish':1,'bearish':0,'neutral':2}[lab]))
            entries.sort(key=lambda x: x[0])
            if len(entries) < 30:
                print("DINO fine-tune skipped: not enough labeled charts.")
            else:
                split = int(0.8 * len(entries))
                train_entries = entries[:split]
                val_entries = entries[split:]

                from torch.utils.data import Dataset, DataLoader
                class ChartClsDataset(Dataset):
                    def __init__(self, items, processor):
                        self.items = items
                        self.processor = processor
                    def __len__(self):
                        return len(self.items)
                    def __getitem__(self, idx):
                        _, img_path, label = self.items[idx]
                        img = Image.open(img_path).convert('RGB').resize((448,448))
                        inputs = self.processor(images=img, return_tensors='pt')
                        # Squeeze batch dim later in collate
                        return {**inputs, 'labels': torch.tensor(label, dtype=torch.long)}

                # Create processor/model head
                from transformers import AutoImageProcessor, Dinov2Model
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                backbone = Dinov2Model.from_pretrained('facebook/dinov2-base')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                backbone = backbone.to(device)

                # Simple linear head on top of pooled features
                num_classes = 3
                class DinoHead(torch.nn.Module):
                    def __init__(self, backbone, hidden=768, ncls=3):
                        super().__init__()
                        self.backbone = backbone
                        self.classifier = torch.nn.Linear(hidden, ncls)
                    def forward(self, pixel_values):
                        out = self.backbone(pixel_values=pixel_values)
                        feats = out.last_hidden_state.mean(dim=1)
                        return self.classifier(feats)

                model = DinoHead(backbone, hidden=768, ncls=num_classes).to(device)

                train_ds = ChartClsDataset(train_entries, processor)
                val_ds = ChartClsDataset(val_entries, processor)

                def collate_fn(batch):
                    # Stack tensors from dicts
                    keys = list(batch[0].keys())
                    pixel_values = torch.cat([b['pixel_values'] for b in batch], dim=0)
                    labels = torch.stack([b['labels'] for b in batch], dim=0)
                    return {'pixel_values': pixel_values, 'labels': labels}

                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
                val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)

                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
                criterion = torch.nn.CrossEntropyLoss()

                best_f1 = 0.0
                from sklearn.metrics import f1_score
                epochs = 6
                for ep in range(epochs):
                    model.train()
                    loss_accum = 0.0
                    for batch in train_loader:
                        pixel_values = batch['pixel_values'].to(device)
                        labels = batch['labels'].to(device)
                        logits = model(pixel_values=pixel_values)
                        loss = criterion(logits, labels)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        loss_accum += loss.item()
                    model.eval()
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for batch in val_loader:
                            pixel_values = batch['pixel_values'].to(device)
                            labels = batch['labels'].cpu().numpy().tolist()
                            preds = model(pixel_values=pixel_values).argmax(dim=1).cpu().numpy().tolist()
                            all_preds += preds; all_labels += labels
                    f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
                    print(f"DINO Epoch {ep+1}/{epochs} | Loss {loss_accum/max(1,len(train_loader)):.4f} | F1 {f1:.4f}")
                    if f1 > best_f1:
                        best_f1 = f1
                        out_dir = BASE_DIR / 'finetuned_dino_crypto'
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # Save processor and state dict
                        torch.save(model.state_dict(), out_dir / 'pytorch_model.bin')
                        processor.save_pretrained(str(out_dir))
                        print(f"Saved finetuned DINO model to {out_dir}")
        else:
            print("DINO fine-tune skipped: metadata.json not found.")
    except Exception as e:
        print(f"DINO fine-tuning error: {e}")


if __name__ == '__main__':
    main()


