# Freedom44 — Hybrid Chart+Timeseries Crypto Trading

> Colab (GPU T4, Python 3.12.11) friendly · Works locally (CPU) via Cursor’s terminal

Freedom44 builds a **hybrid model** that blends:
- **Chart vision** (DINO/CLIP embeddings of rendered price windows, optional YOLO pseudo‑labels)
- **Time‑series features** (microstructure, volatility/HAR‑RV, regime, basis/funding)
- **Leakage‑safe validation** (purged K‑fold + embargo)
- **Risk controls** (adaptive time‑series conformal bounds, deflated Sharpe & PBO overfit guards)
- **(Optional)** survival timing (Mamba SSM or fallback encoder)

---

## TL;DR quickstart

### A. Run on **Google Colab** (GPU T4) — recommended
1. Upload the repo zip (or `git clone`) into Colab `/content`.
2. Open a Python 3.12 runtime with **T4 GPU**.
3. Install deps and run **Part 0** → **Part 1B**:
   ```bash
   %%bash
   pip -q install --upgrade pip
   pip -q install -r /content/freedom44/requirements.txt
   pip -q install open-clip-torch ccxt yfinance
   export HYBRID_WORKSPACE="/content/hybrid_workspace"
   python /content/freedom44/part0_setup_data_gen.py      --symbols "BTC-USD,ETH-USD"      --start "2023-01-01" --end "2023-03-15"      --rep "dino" --outdir "$HYBRID_WORKSPACE" --seed 0
   export HYBRID_BASE_DIR="$HYBRID_WORKSPACE"
   python /content/freedom44/part1b_two_stage_hybrid.py --embargo 10
   ```
   Optional: tune (`part1c_optuna_tune.py`) or run survival (`part2_survival_mamba.py`, needs `mamba-ssm`).

### B. Run in **Cursor** (local CPU)
> Great for editing/debugging; use Colab for the heavier GPU steps.

1. **Install Python 3.12.11** (via pyenv or python.org).
2. **Open the repo in Cursor** → open the **Terminal** (View → Terminal).
3. **Create a venv** and install deps:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install open-clip-torch ccxt yfinance
   ```
4. **Run a tiny CPU smoke test** (short range to stay fast):
   ```bash
   export HYBRID_WORKSPACE="$(pwd)/hybrid_workspace"
   python part0_setup_data_gen.py --symbols "BTC-USD" --start "2023-01-01" --end "2023-02-15" --rep "dino" --outdir "$HYBRID_WORKSPACE" --seed 0
   export HYBRID_BASE_DIR="$HYBRID_WORKSPACE"
   python part1b_two_stage_hybrid.py --embargo 10
   ```
   Windows (PowerShell) replace `export` with:
   ```powershell
   $env:HYBRID_WORKSPACE="$pwd\hybrid_workspace"
   $env:HYBRID_BASE_DIR=$env:HYBRID_WORKSPACE
   ```

---

## File map (selected)

- `part0_setup_data_gen.py` — ingest OHLCV → ICT indicators → render chart images → YOLO pseudo‑labels → DINO/CLIP embeddings → write workspace.
- `part1_hybrid_train_backtest.py` — single‑stage hybrid + calibrated backtest.
- `part1b_two_stage_hybrid.py` — two‑stage movement→direction with gating; improved backtest.
- `part1c_optuna_tune.py` — Optuna tuning of Stage A/B + gates.
- `part2_survival_mamba.py` — survival timing (Mamba SSM optional).
- `cv/purged.py` — purged K‑fold with embargo (anti‑leakage).
- `risk/conformal_adaptive.py` — adaptive TS conformal bounds.
- `labels/*` — triple‑barrier and survival labels.
- `portfolio/allocate.py` — HRP allocation + leverage‑down.
- `selection/overfit_guard.py` — deflated Sharpe + PBO.
- `conf/experiment.yaml` — feature/model/conformal/CV toggles.

---

## Configuration

- **Workspace**: set `HYBRID_WORKSPACE` (Part 0 writer) and `HYBRID_BASE_DIR` (Parts 1/2 reader). Default is `/content/hybrid_workspace` on Colab.
- **Symbols & dates**: Part 0 supports comma‑separated tickers; start/end bound the sample and runtime.
- **Features**: toggle in `conf/experiment.yaml` (microstructure, HAR, regime, basis/funding).
- **Gating thresholds**: in `part1b_two_stage_hybrid.py` (`m_conf`, direction margin).
- **Labels**: tweak triple‑barrier multipliers and survival horizons in `labels/*`.

---

## Acceptance checks

- **No NaNs** in core OHLCV post‑preprocessing.
- **Leakage control** enabled (purged + embargo).
- **Backtest sanity**: non‑trivial trades, Sharpe > 0 on validation tail, reasonable MaxDD.
- **Overfit guards** OK: deflated Sharpe ≥ ~0.10, PBO ≤ ~0.40.
- **Conformal** coverage ≈ target (~0.85) and reduces tail losses with gates.

---

## Troubleshooting

- **Missing packages**: add `open-clip-torch`, `ccxt`, `yfinance` (some code paths import them).
- **signatory (optional)**: often fails on Python 3.12; safe to skip (features degrade gracefully).
- **sklearn bound**: if `>=1.6` isn’t available on your image, use `scikit-learn~=1.5` temporarily.
- **GPU OOM**: narrow date range or reduce image size; T4 is recommended.

---

## License

TBD by repository owner.
