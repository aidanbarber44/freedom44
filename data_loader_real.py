import os
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Feature Engineering Imports (from your project structure) ---
from .features.microstructure import add_microstructure
from .features.volatility import add_har_rv
from .features.regime import add_regime

def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Computes Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

def _find_and_load_asset_csv(asset: str, data_dir: Path) -> pd.DataFrame | None:
    """
    Finds and loads the specific 1-minute CSV for a given asset from the extracted data directory.
    """
    # Kraken uses XBT for Bitcoin, so we must search for that ticker specifically.
    search_asset = "XBTUSD" if asset == "BTCUSD" else asset
    
    target_filename = f"{search_asset}_1.csv"
    
    # Use rglob to search recursively, which is robust to nested directories.
    potential_paths = list(data_dir.rglob(target_filename))
    
    if not potential_paths:
        # Fallback for XRP which sometimes uses a different convention
        if search_asset == "XRPUSD":
            potential_paths = list(data_dir.rglob("XRPUSD_1.csv"))

    if not potential_paths:
        print(f"Warning: Data file not found for {asset} (searched for {target_filename})")
        return None

    csv_path = potential_paths[0] # Take the first match
    
    try:
        df = pd.read_csv(
            csv_path,
            names=["time", "open", "high", "low", "close", "volume", "trades"],
            skiprows=1,
            low_memory=False
        )
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    except Exception as e:
        print(f"Error reading {csv_path} for asset {asset}: {e}")
        return None


def load_features_for_symbols(symbols: list, conf: dict) -> dict:
    """
    Main function to load, process, and feature-engineer 4h data for all specified symbols.
    """
    drive_zip_path = Path('/content/drive/MyDrive/Kraken_OHLCVT.zip')
    extract_path = Path('/content/Kraken_Data')
    
    # 1. Unzip data if not already present
    if not extract_path.exists() or not any(extract_path.iterdir()):
        print(f"Extracting {drive_zip_path} to {extract_path}...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(drive_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")

    all_features = {}
    
    # 2. Process each symbol
    for symbol in tqdm(symbols, desc="Processing Real Asset Data"):
        df_1m = _find_and_load_asset_csv(symbol, extract_path)

        if df_1m is None or df_1m.empty:
            continue

        # 3. Resample to 4-hour timeframe
        # --- FIX: Changed '4H' to '4h' to avoid FutureWarning ---
        df_4h = df_1m.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if df_4h.empty:
            continue

        # 4. Feature Engineering
        df_4h['atr'] = _compute_atr(df_4h)
        
        # Add features from your project's modules
        micro_feats = add_microstructure(df_4h, price='close', high='high', low='low', vol='volume')
        har_feats = add_har_rv(df_4h['close'].pct_change())
        regime_feats = add_regime(df_4h['close'])
        
        # Combine all features
        df_featured = pd.concat([df_4h, micro_feats, har_feats, regime_feats], axis=1)
        
        # Clean up any NaNs produced by rolling windows
        df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_featured.dropna(inplace=True)
        
        if not df_featured.empty:
            all_features[symbol] = df_featured

    print(f"Successfully loaded and processed real data for {len(all_features)} assets.")
    return all_features

