# setup_project_structure.py
import os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).parent

PKG_DIRS = ["conf","utils","features","labels","models","experts","risk","portfolio","selection","reports","cv"]

# mapping for accidental "name:subname.py" uploads -> proper paths
MOVE_MAP = {
    "cv:purged.py":               "cv/purged.py",
    "experts:regime_router.py":   "experts/regime_router.py",
    "features:funding_basis.py":  "features/funding_basis.py",
    "features:microstructure.py": "features/microstructure.py",
    "features:regime.py":         "features/regime.py",
    "features:volatility.py":     "features/volatility.py",
    "labels:tripple_barrier.py":  "labels/triple_barrier.py",  # fix spelling
    "models:gbm.py":              "models/gbm.py",
    "models:stacking.py":         "models/stacking.py",
    "portfolio:allocate.py":      "portfolio/allocate.py",
    "risk:conformal.py":          "risk/conformal.py",
    "selection:overfit_guard.py": "selection/overfit_guard.py",
    "utils:seed.py":              "utils/seed.py",
    "conf:experiment.yaml":       "conf/experiment.yaml",
}

def ensure_dirs():
    for d in PKG_DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        initf = p / "__init__.py"
        if d not in ("reports","conf") and not initf.exists():
            initf.write_text("")  # make it a package (not needed for conf/reports)

def move_colon_files():
    moved = []
    for src, dst in MOVE_MAP.items():
        srcp = ROOT / src
        dstp = ROOT / dst
        if srcp.exists():
            dstp.parent.mkdir(parents=True, exist_ok=True)
            if not dstp.exists():
                shutil.move(str(srcp), str(dstp))
                moved.append((src, dst))
    return moved

def main():
    ensure_dirs()
    moved = move_colon_files()
    print("Folders ready.")
    if moved:
        print("Moved files:")
        for s, d in moved:
            print(f"  {s} -> {d}")
    else:
        print("No colon-named files found to move (that’s okay).")
    # quick report of key expected files:
    expected = [
        "conf/experiment.yaml",
        "utils/seed.py",
        "cv/purged.py",
        "features/microstructure.py",
        "features/volatility.py",
        "features/regime.py",
        "features/funding_basis.py",
        "labels/triple_barrier.py",
        "models/gbm.py",
        "models/stacking.py",
        "experts/regime_router.py",
        "portfolio/allocate.py",
        "part0_setup_data_gen.py",
        "part1_hybrid_train_backtest.py",
        "part1b_two_stage_hybrid.py",
        "part1c_optuna_tune.py",
    ]
    missing = [e for e in expected if not (ROOT / e).exists()]
    if missing:
        print("\n⚠️ Missing (check names/paths):")
        for m in missing: print(" ", m)
    else:
        print("\nAll key files present. ✅")

if __name__ == "__main__":
    sys.exit(main())
