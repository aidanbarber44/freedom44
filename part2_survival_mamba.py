import os, math, numpy as np, pandas as pd, torch, logging
from torch.utils.data import DataLoader
from typing import Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Robust local imports mirroring Part 1 style
try:
    from utils.seed import set_all_seeds  # type: ignore
except Exception:
    def set_all_seeds(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

try:
    from labels.triple_barrier_survival import build_survival_targets  # type: ignore
except Exception:
    from freedom44.labels.triple_barrier_survival import build_survival_targets  # type: ignore
try:
    from features.sequence_builder import SequenceSurvivalDataset  # type: ignore
except Exception:
    from freedom44.features.sequence_builder import SequenceSurvivalDataset  # type: ignore
try:
    from models.mamba_survival import SurvivalMamba, deephit_loss  # type: ignore
except Exception:
    from freedom44.models.mamba_survival import SurvivalMamba, deephit_loss  # type: ignore
try:
    from cv.purged import purged_time_splits  # type: ignore
except Exception:
    from freedom44.cv.purged import purged_time_splits  # type: ignore
try:
    from risk.conformal_adaptive import AdaptiveTSConformal  # type: ignore
except Exception:
    from freedom44.risk.conformal_adaptive import AdaptiveTSConformal  # type: ignore
try:
    from selection.overfit_guard import guard_or_fail  # type: ignore
except Exception:
    from freedom44.selection.overfit_guard import guard_or_fail  # type: ignore
# --- MODIFICATION: Import the real data loader ---
try:
    from freedom44.data_loader_real import load_features_for_symbols
except ImportError:
    print("WARNING: Could not import real data loader. Falling back to synthetic data.")
    # This fallback ensures the script can still run if the new file isn't present
    def load_features_for_symbols(symbols, conf):
        idx = pd.date_range("2022-01-01", periods=10000, freq="H")
        feats = {}
        for s in symbols:
            close = pd.Series(np.cumsum(np.random.randn(len(idx)) * 0.1) + 100.0, index=idx)
            atr = close.rolling(14).std().fillna(close.std() * 0.1)
            f = pd.DataFrame({"close": close, "atr": atr})
            f["ret1"] = f["close"].pct_change().fillna(0.0)
            f["vol14"] = f["ret1"].rolling(14).std().fillna(0.0)
            feats[s] = f
        return feats


# Optional Colab nicety for T4 GPUs
try:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
except Exception:
    logging.debug("Torch precision tweak not applied")

DEFAULT_CONFIG = {
    "data": {
        "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
        "interval": "1h",
        "lookback_bars": 240,
        "horizon_bars": 12,
        "train_start": "2019-01-01",
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2024-06-30",
    },
    "model": {
        "encoder": "gru",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "competing_risks": True,
    },
    "training": {
        "batch_size": 256,
        "epochs": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "purged_folds": 3,
        "embargo_bars": 24,
    },
    "execution": {"tp_sl": {"atr_mult_tp": 2.5, "atr_mult_sl": 0.75}},
    "conformal": {"enabled": True, "target_coverage": 0.85, "window": 2000},
    "cv": {"outer_folds": 3, "inner_folds": 2},
}

def _normalize_config_keys(cfg: dict) -> dict:
    # Accept both 'model' and 'modeling'
    if "model" not in cfg and "modeling" in cfg:
        cfg["model"] = cfg["modeling"]
        logging.warning("Config uses 'modeling'; normalizing to 'model'")
    if "modeling" not in cfg and "model" in cfg:
        cfg["modeling"] = cfg["model"]
    # Ensure 'execution'
    cfg.setdefault("execution", {"tp_sl": {"atr_mult_tp": 2.5, "atr_mult_sl": 0.75}})
    # Ensure 'conformal'
    cfg.setdefault("conformal", {"enabled": True, "target_coverage": 0.85, "window": 2000})
    # Ensure 'cv'
    cfg.setdefault("cv", {"outer_folds": 3, "inner_folds": 2})
    return cfg


def load_config(path_default: str) -> Dict:
    paths = [path_default, os.path.join(os.path.dirname(__file__), 'conf', 'experiment.yaml')]
    for p in paths:
        try:
            if os.path.exists(p) and yaml is not None:
                with open(p, 'r') as f:
                    try:
                        cfg = yaml.safe_load(f)
                        return _normalize_config_keys(cfg)
                    except Exception:
                        logging.exception("Failed to parse config at %s; skipping", p)
                        continue
        except Exception:
            continue
    logging.warning("No valid YAML found; using DEFAULT_CONFIG")
    return _normalize_config_keys(DEFAULT_CONFIG)

def train_one_asset(df: pd.DataFrame, conf: Dict, device: str = "cpu") -> Dict:
    modeling = conf.get("modeling", {})
    W = int(modeling.get("sequence_window", 128))
    K = int(modeling.get("survival_bins", 32))
    mode = str(modeling.get("survival_bin_mode", "bars"))
    tp_mult = float(conf.get("execution", {}).get("tp_sl", {}).get("atr_mult_tp", 2.5))
    sl_mult = float(conf.get("execution", {}).get("tp_sl", {}).get("atr_mult_sl", 0.75))
    horizon = int(conf.get("data", {}).get("horizon_bars", 12))

    targets = build_survival_targets(df, tp_mult=tp_mult, sl_mult=sl_mult, horizon_max=horizon, mode=mode, bins=K)
    features_df = df.drop(columns=["close"])  # full; dataset aligns indices
    ds = SequenceSurvivalDataset(features_df, targets, window=W, signature_cfg=modeling.get("signature_features", {}))
    
    # Handle case where dataset is too small to train
    if len(ds) == 0:
        logging.warning(f"Dataset for asset is empty after processing. Skipping training.")
        return {"model": None, "dataset": None}
        
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

    input_dim = ds.num_features
    model = SurvivalMamba(
        input_dim=input_dim,
        hidden_size=int(modeling.get("hidden_size", 256)),
        num_layers=int(modeling.get("num_layers", 4)),
        K=K,
        dropout=float(modeling.get("dropout", 0.1)),
        encoder=str(modeling.get("encoder", "mamba")),
        use_movement_head=bool(modeling.get("use_movement_head", True)),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Increased epochs for meaningful training on real data
    for epoch in range(conf.get("training", {}).get("epochs", 8)):
        model.train()
        for x_seq, meta, y in loader:
            x_seq = x_seq.to(device)
            out = model(x_seq)
            tgt = {
                "risk": y["risk"].to(device).long(),
                "time_bin": y["time_bin"].to(device).long(),
            }
            loss = deephit_loss(out, tgt, use_movement_head=modeling.get("use_movement_head", True))
            opt.zero_grad(); loss.backward(); opt.step()
    return {"model": model, "dataset": ds}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=str, default=os.path.join(os.path.dirname(__file__), "conf", "experiment.yaml"))
    p.add_argument("--assets", type=str, default="BTCUSD,ETHUSD,SOLUSD,ADAUSD,LTCUSD,XRPUSD,AVAXUSD,LINKUSD,DOTUSD,ATOMUSD,SUIUSD,UNIUSD")
    p.add_argument("--save", type=str, default="hybrid_workspace/hybrid_models_survival")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    conf = load_config(args.conf)
    set_all_seeds(42)
    os.makedirs(args.save, exist_ok=True)

    symbols = [s.strip() for s in args.assets.split(",") if s.strip()]
    
    # --- MODIFICATION: This now calls the real data loader ---
    print("Loading real data from Google Drive...")
    feats = load_features_for_symbols(symbols, conf)

    results = {}
    # --- NEW: Create a directory for saved models ---
    models_save_dir = os.path.join(args.save, "trained_model_states")
    os.makedirs(models_save_dir, exist_ok=True)

    for s in symbols:
        if s not in feats or feats[s].empty:
            print(f"No data for {s}, skipping training.")
            continue
        print(f"Training survival model for {s}...")
        out = train_one_asset(feats[s], conf, device=args.device)
        
        # We only record success if a model was actually trained
        if out.get("model"):
            results[s] = {"trained": True}
            # --- NEW: Save the trained model's state dictionary ---
            model_save_path = os.path.join(models_save_dir, f"{s}_model.pth")
            torch.save(out["model"].state_dict(), model_save_path)
            print(f"Saved model for {s} to {model_save_path}")
        else:
            results[s] = {"trained": False}


    pd.DataFrame.from_dict(results, orient="index").to_csv(os.path.join(args.save, "results_summary.csv"))

if __name__ == "__main__":
    main()

