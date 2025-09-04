import os, sys, io, json, math, time, hashlib, random, logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image


def set_all_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


# Detector classes constant
CLASS_NAMES = [
    "fvg_bull","fvg_bear","ob_bull","ob_bear","bos_bull","bos_bear","choch_bull","choch_bear"
]


def _utcify(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def _hash_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns: timestamp, open, high, low, close, volume
    # normalize timestamp
    ts_col = None
    for c in ['timestamp', 'time', 'ts', 'date']:
        if c in df.columns:
            ts_col = c
            break
    assert ts_col is not None, f"No timestamp column in {csv_path}"
    df = df.rename(columns={ts_col: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    df = df.set_index('timestamp').sort_index()
    # basic column normalization
    ren = { 'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume' }
    df = df.rename(columns=ren)
    keep = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    return df[keep]


# ====== New helpers for event-centered crops and balancing ======
RARE_CLASSES = {"ob_bull","ob_bear","bos_bull","bos_bear","choch_bull","choch_bear"}
SCALES = [1.6, 2.0, 2.4]
TARGET_PER_CLASS = 25000

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _yolo_txt_to_boxes(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                parts = parts[:5]
            cls, xc, yc, w, h = parts
            xc, yc, w, h = map(float, (xc, yc, w, h))
            bx = (xc - w/2) * img_w
            by = (yc - h/2) * img_h
            ex = (xc + w/2) * img_w
            ey = (yc + h/2) * img_h
            boxes.append((int(cls), bx, by, ex, ey))
    return boxes

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def _remap_label_to_crop(bx, by, ex, ey, crop_xyxy, img_w, img_h):
    x1,y1,x2,y2 = crop_xyxy
    ix1, iy1 = max(bx,x1), max(by,y1)
    ix2, iy2 = min(ex,x2), min(ey,y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    cw, ch = x2-x1, y2-y1
    cx = (ix1+ix2)/2 - x1
    cy = (iy1+iy2)/2 - y1
    w  = (ix2-ix1)
    h  = (iy2-iy1)
    return (cx/cw, cy/ch, w/cw, h/ch)

def generate_event_crops(yolo_root: Path, out_root: Path, rng: random.Random, target_per_class: int = TARGET_PER_CLASS) -> pd.DataFrame:
    img_dir = yolo_root / "images"
    lab_dir = yolo_root / "labels"
    out_img = out_root / "images_crops"
    out_lab = out_root / "labels_crops"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    cls_counts = {}
    for lp in lab_dir.rglob("*.txt"):
        imgp = img_dir / lp.with_suffix(".jpg").name
        if not imgp.exists():
            imgp = img_dir / lp.with_suffix(".png").name
        if not imgp.exists():
            continue
        with Image.open(imgp) as im:
            w,h = im.size
        for c, *_ in _yolo_txt_to_boxes(lp, w, h):
            cls_counts[c] = cls_counts.get(c, 0) + 1

    max_cls = max(cls_counts.values() or [1])
    oversample = {c: max(1, math.ceil(min(3.0, max_cls / max(v,1)))) for c, v in cls_counts.items()}

    rows = []
    for lp in lab_dir.rglob("*.txt"):
        imgp = img_dir / lp.with_suffix(".jpg").name
        if not imgp.exists():
            imgp = img_dir / lp.with_suffix(".png").name
        if not imgp.exists():
            continue
        im = Image.open(imgp).convert("RGB")
        w,h = im.size
        boxes = _yolo_txt_to_boxes(lp, w, h)
        if not boxes:
            continue

        for (cls, bx, by, ex, ey) in boxes:
            k = oversample.get(cls, 1)
            for _ in range(k):
                scale = rng.choice(SCALES)
                cx, cy = (bx+ex)/2, (by+ey)/2
                bw, bh = (ex-bx)*scale, (ey-by)*scale
                x1 = _clip(cx - bw/2, 0, w-1)
                y1 = _clip(cy - bh/2, 0, h-1)
                x2 = _clip(cx + bw/2, 0, w-1)
                y2 = _clip(cy + bh/2, 0, h-1)
                crop = im.crop((x1,y1,x2,y2))
                remapped = []
                for (c2, bx2, by2, ex2, ey2) in boxes:
                    r = _remap_label_to_crop(bx2, by2, ex2, ey2, (x1,y1,x2,y2), w, h)
                    if r is not None:
                        remapped.append((c2, *r))
                if not remapped:
                    continue

                stem = f"{imgp.stem}_{int(cx)}_{int(cy)}_{int(scale*100)}_{rng.randint(0,10**9):09d}"
                out_img_p = out_img / f"{stem}.jpg"
                out_lab_p = out_lab / f"{stem}.txt"
                crop.save(out_img_p, quality=92, subsampling=1)
                with open(out_lab_p, "w") as f:
                    for (c2, xc, yc, ww, hh) in remapped:
                        f.write(f"{c2} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                sh = _sha256_file(out_img_p)
                rows.append({
                    "image_path": str(out_img_p),
                    "label_path": str(out_lab_p),
                    "sha256": sh,
                    "img_w": crop.size[0], "img_h": crop.size[1],
                    "src_image": str(imgp), "src_label": str(lp),
                    "center_x": float(cx), "center_y": float(cy), "scale": float(scale),
                    "view": "crop"
                })

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        def _first_cls(path):
            with open(path) as f:
                line = f.readline().strip().split()
                return int(line[0]) if line else -1
        manifest["first_class"] = manifest["label_path"].map(_first_cls)
        capped = []
        for c, g in manifest.groupby("first_class"):
            n = len(g)
            cap = TARGET_PER_CLASS if CLASS_NAMES[c] in RARE_CLASSES else int(TARGET_PER_CLASS * 1.2)
            if n > cap:
                capped.append(g.sample(cap, random_state=0))
            else:
                capped.append(g)
        manifest = pd.concat(capped).sample(frac=1.0, random_state=0).reset_index(drop=True)
    return manifest



def _append_ccxt(symbol: str, since: pd.Timestamp, df: pd.DataFrame, retries: int = 3) -> pd.DataFrame:
    try:
        import ccxt
    except Exception:
        return df
    ex = ccxt.binance()
    ms_since = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    cur = ms_since
    for _ in range(retries):
        try:
            # binance spot: symbol like BTC/USDT; convert ADAUSD -> ADA/USDT best effort
            s = symbol.replace('USD', '/USDT') if '/' not in symbol else symbol
            ohlcv = ex.fetch_ohlcv(s, timeframe='1m', since=cur, limit=1000)
            if not ohlcv:
                break
            out.extend(ohlcv)
            cur = ohlcv[-1][0] + 60_000
            time.sleep(ex.rateLimit / 1000.0)
        except Exception:
            time.sleep(1.0)
            continue
    if not out:
        return df
    d2 = pd.DataFrame(out, columns=['timestamp','open','high','low','close','volume'])
    d2['timestamp'] = pd.to_datetime(d2['timestamp'], unit='ms', utc=True).dt.tz_convert(None)
    d2 = d2.set_index('timestamp').sort_index()
    merged = pd.concat([df, d2], axis=0)
    merged = merged[~merged.index.duplicated(keep='last')].sort_index()
    return merged


def _ffill_small_gaps(df: pd.DataFrame, max_gap: int = 2) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    # assume 1m bars; detect missing minutes and ffill up to max_gap, record longer gaps
    info = { 'long_gaps': [] }
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='1min')
    df_reidx = df.reindex(full_range)
    na_groups = df_reidx['close'].isna().astype(int).groupby((~df_reidx['close'].isna()).cumsum()).sum()
    # ffill up to max_gap
    df_ff = df_reidx.copy()
    df_ff[['open','high','low','close','volume']] = df_ff[['open','high','low','close','volume']].fillna(method='ffill', limit=max_gap)
    # mark remaining gaps
    rem = df_ff['close'].isna()
    if rem.any():
        # record ranges
        gaps = []
        cur = None
        for t, isna in rem.items():
            if isna and cur is None:
                cur = [t, t]
            elif isna and cur is not None:
                cur[1] = t
            elif (not isna) and cur is not None:
                gaps.append((cur[0], cur[1])); cur=None
        if cur is not None: gaps.append((cur[0], cur[1]))
        info['long_gaps'] = [f"{a} -> {b}" for a,b in gaps]
        # keep NaNs to track integrity, will be dropped after resample checks
    df_ff = df_ff.astype(float)
    return df_ff, info


def _resample(df1m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _agg(x):
        return pd.Series({
            'open': x['open'].iloc[0],
            'high': x['high'].max(),
            'low': x['low'].min(),
            'close': x['close'].iloc[-1],
            'volume': x['volume'].sum(),
        })
    df4h = df1m.resample('4H', label='right', closed='right').apply(_agg).dropna(how='any')
    df1d = df1m.resample('1D', label='right', closed='right').apply(_agg).dropna(how='any')
    return df4h, df1d


def _windows(df: pd.DataFrame, lookback: int, horizon: int, stride: int, embargo: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    idx = df.index
    out = []
    i = lookback
    n = len(idx)
    while i + horizon < n:
        start = idx[i - lookback]
        end = idx[i]
        fut_end = idx[i + horizon]
        out.append((start, end, fut_end))
        i += stride
        i += embargo
    return out


def _label_change(df: pd.DataFrame, t_end: pd.Timestamp, t_future: pd.Timestamp, tp: float = 0.01, sl: float = 0.005) -> str:
    p0 = float(df.loc[t_end, 'close'])
    path = df.loc[t_end:t_future, 'close'].astype(float)
    up = (path >= p0 * (1 + tp)).any()
    dn = (path <= p0 * (1 - sl)).any()
    if up and not dn: return 'bullish'
    if dn and not up: return 'bearish'
    return 'neutral'


def _balanced_indices(labels: List[str], max_per_class: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    idx_by = {}
    for i, lab in enumerate(labels):
        idx_by.setdefault(lab, []).append(i)
    keep = []
    for lab, idxs in idx_by.items():
        if len(idxs) > max_per_class:
            keep.extend(list(rng.choice(idxs, size=max_per_class, replace=False)))
        else:
            keep.extend(idxs)
    keep.sort()
    return keep


def _sha256_image_bytes(img_path: Path) -> str:
    return _hash_file(img_path)


def build_all(symbols: List[str], start: str, end: str, outdir: str, seed: int = 0, use_drive: bool = False) -> Dict[str, str]:
    set_all_seeds(seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    out = {}
    base = Path(outdir)
    ds_dir = base / 'datasets' / 'chart_dataset'
    yolo_dir = base / 'datasets' / 'yolo_ict_det'
    _safe_mkdir(ds_dir); _safe_mkdir(yolo_dir)
    img_dir = yolo_dir / 'images'; lbl_dir = yolo_dir / 'labels'
    _safe_mkdir(img_dir); _safe_mkdir(lbl_dir)

    # ingest symbols
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    manifest_rows_full = []
    meta = { 'symbols': {}, 'ranges': {}, 'long_gaps': {}, 'counts': {}, 'splits': {} }

    for sym in symbols:
        # CSV path convention: {sym}_1m.csv under outdir/datasets/raw
        raw_dir = base / 'datasets' / 'raw'
        csv_path = raw_dir / f"{sym}_1m.csv"
        df = None
        if csv_path.exists():
            df = _read_csv_ohlcv(csv_path)
        else:
            logging.warning(f"CSV not found for {sym}: {csv_path}. Proceeding with CCXT only may be slow.")
            df = pd.DataFrame(columns=['open','high','low','close','volume'])
            # need a minimal index to append
            df.index = pd.date_range(start_ts, periods=1, freq='1min')
        df = _utcify(df)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        # CCXT append
        df = _append_ccxt(sym, df.index.max() if len(df) else start_ts, df)
        df, info = _ffill_small_gaps(df)
        meta['long_gaps'][sym] = info.get('long_gaps', [])
        # resample
        df4h, _ = _resample(df)
        assert not df4h.isna().any().any(), f"NaNs after resample for {sym}"
        rng = np.random.RandomState(seed)
        # windows and labels
        lookback, horizon, stride, embargo = 240, 12, 5, 2
        wins = _windows(df4h, lookback, horizon, stride, embargo)
        labels = [_label_change(df4h, a[1], a[2]) for a in wins]
        # balance
        counts = {k: labels.count(k) for k in ['bullish','bearish','neutral']}
        max_per = int(np.median(list(counts.values()))) if counts else 0
        sel = _balanced_indices(labels, max_per_class=max_per or 0, seed=seed)
        wins_b = [wins[i] for i in sel]
        labels_b = [labels[i] for i in sel]
        # splits deterministic by time
        times = [w[1] for w in wins_b]
        order = np.argsort(times)
        cut = int(0.8 * len(order))
        train_idx = set(order[:cut]); val_idx = set(order[cut:])
        # unit checks
        assert len(train_idx.intersection(val_idx)) == 0
        if times:
            meta['ranges'][sym] = f"{min(times)} -> {max(times)}"
        meta['counts'][sym] = { 'total': len(labels_b), 'bull': labels_b.count('bullish'), 'bear': labels_b.count('bearish'), 'neu': labels_b.count('neutral') }
        # images/labels: we will write placeholder crops (full-frame as one crop) to keep pipeline consistent
        for i, ((t0, t1, t2), lab) in enumerate(zip(wins_b, labels_b)):
            fname = f"{sym}_{pd.Timestamp(t1).strftime('%Y%m%d%H%M%S')}_{i}.jpg"
            img_path = img_dir / ('train' if i in train_idx else 'val') / fname
            img_path.parent.mkdir(parents=True, exist_ok=True)
            # save a tiny 1x1 image placeholder (actual chart generation happens in Part0 script if enabled)
            try:
                from PIL import Image
                Image.new('RGB', (64, 64), color=(0, 0, 0)).save(str(img_path))
            except Exception:
                pass
            lbl_path = lbl_dir / img_path.parent.name / (img_path.stem + '.txt')
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            # no boxes by default; detector training expects files; write empty label file
            lbl_path.write_text("")
            manifest_rows_full.append({
                'image_path': str(img_path), 'symbol': sym, 'ts': str(t1), 'frame_size': '64x64',
                'crop_meta': json.dumps({'type': 'full'}), 'label_counts': json.dumps({'bull':0,'bear':0,'neu':0}),
                'sha256': _sha256_image_bytes(img_path), 'split': img_path.parent.name,
            })
        meta['symbols'][sym] = True

    # Baseline full-frame manifest
    manifest_full = pd.DataFrame(manifest_rows_full)
    if not manifest_full.empty:
        manifest_full['view'] = 'full'
        # ensure ts as Timestamp UTC-naive
        manifest_full['ts'] = pd.to_datetime(manifest_full['ts'], utc=True).dt.tz_convert(None)

    # Generate multi-crops from any existing labels under yolo_dir
    rng = random.Random(seed)
    crops_manifest = generate_event_crops(yolo_dir, yolo_dir, rng)
    if not crops_manifest.empty:
        # try to infer symbol and ts from src image stem pattern
        def _infer_symbol(path: str) -> str:
            stem = Path(path).stem
            return stem.split('_')[0]
        def _infer_ts(path: str):
            stem = Path(path).stem
            parts = stem.split('_')
            for p in parts:
                if len(p) >= 14 and p.isdigit():
                    try:
                        return pd.to_datetime(p, format='%Y%m%d%H%M%S', utc=True).tz_convert(None)
                    except Exception:
                        continue
            return pd.NaT
        crops_manifest['symbol'] = crops_manifest.get('symbol', pd.Series([None]*len(crops_manifest)))
        crops_manifest['symbol'] = crops_manifest['symbol'].fillna(crops_manifest['src_image'].map(_infer_symbol))
        crops_manifest['ts'] = crops_manifest.get('ts', pd.Series([None]*len(crops_manifest)))
        crops_manifest['ts'] = pd.to_datetime(crops_manifest['ts'], errors='coerce', utc=True).dt.tz_convert(None)
        crops_manifest['ts'] = crops_manifest['ts'].fillna(crops_manifest['src_image'].map(_infer_ts))

    # Merge manifests and set split deterministically (time-ordered with embargo)
    merged = pd.concat([manifest_full[['image_path']].assign(label_path='') if not manifest_full.empty else pd.DataFrame(columns=['image_path','label_path']),
                        crops_manifest[['image_path','label_path']] if not crops_manifest.empty else pd.DataFrame(columns=['image_path','label_path'])],
                       axis=0, ignore_index=True)
    # Attach metadata columns
    if not manifest_full.empty:
        md_full = manifest_full.set_index('image_path')
        merged['symbol'] = merged['image_path'].map(md_full['symbol'])
        merged['ts'] = merged['image_path'].map(md_full['ts'])
        merged['view'] = merged['image_path'].map(md_full['view']).fillna('crop')
    if not crops_manifest.empty:
        md_crop = crops_manifest.set_index('image_path')
        merged['symbol'] = merged['symbol'].fillna(merged['image_path'].map(md_crop['symbol']))
        merged['ts'] = merged['ts'].fillna(merged['image_path'].map(md_crop['ts']))
        merged['view'] = merged['view'].fillna(merged['image_path'].map(md_crop['view']))
    # sha256
    merged['sha256'] = merged['image_path'].map(lambda p: _sha256_file(Path(p)) if Path(p).exists() else '')
    # time split with embargo
    def make_time_splits(manifest: pd.DataFrame, embargo: int = 10) -> pd.DataFrame:
        m = manifest.copy()
        m['ts'] = pd.to_datetime(m['ts'], utc=True).dt.tz_convert(None)
        m = m.sort_values('ts').reset_index(drop=True)
        N = len(m)
        train_end = int(N * 0.80)
        val_start = min(N-1, train_end + embargo)
        m.loc[:train_end, 'split'] = 'train'
        m.loc[val_start:, 'split'] = 'val'
        if not m.empty:
            assert m.loc[m['split']=='train','ts'].max() < m.loc[m['split']=='val','ts'].min()
        return m
    merged = make_time_splits(merged, embargo=10)

    # Write YOLO lists
    def write_yolo_lists(dataset_root: Path, manifest: pd.DataFrame) -> dict:
        train_paths = manifest.loc[manifest['split']=='train','image_path'].tolist()
        val_paths   = manifest.loc[manifest['split']=='val','image_path'].tolist()
        (dataset_root / 'train.txt').write_text("\n".join(train_paths))
        (dataset_root / 'val.txt').write_text("\n".join(val_paths))
        return {'train_list': str(dataset_root / 'train.txt'), 'val_list': str(dataset_root / 'val.txt')}
    lists = write_yolo_lists(yolo_dir, merged)

    # write manifests
    manifest_csv = ds_dir / 'images_manifest.csv'
    merged.to_csv(manifest_csv, index=False)
    meta_path = ds_dir / 'metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2))

    # checksums
    def write_checksums(outdir: Path, files: List[str]):
        payload = {}
        for fp in files:
            p = Path(fp)
            if p.exists():
                payload[str(p)] = _sha256_file(p)
        (outdir / 'checksums.json').write_text(json.dumps(payload, indent=2))
    files = [str(manifest_csv), str(meta_path), lists['train_list'], lists['val_list']]
    clip_parq = ds_dir / 'clip_embeddings.parquet'
    if clip_parq.exists():
        files.append(str(clip_parq))
    write_checksums(ds_dir, files)

    # Print sanity summary
    len_full = 0 if manifest_full is None or manifest_full.empty else len(manifest_full)
    len_crops = 0 if crops_manifest is None or crops_manifest.empty else len(crops_manifest)
    len_total = len(merged)
    n_train = (merged['split'] == 'train').sum()
    n_val = (merged['split'] == 'val').sum()
    ts_min = str(pd.to_datetime(merged['ts'], utc=True).min().tz_convert(None)) if not merged.empty else ''
    ts_max = str(pd.to_datetime(merged['ts'], utc=True).max().tz_convert(None)) if not merged.empty else ''
    embargo = 10
    print(f"Crops: {len_crops} | Full frames: {len_full} | Total: {len_total}")
    print(f"Train images: {n_train} | Val images: {n_val} | Embargo: {embargo}")
    print(f"Time range: {ts_min} -> {ts_max} (UTC)")

    out['metadata_json'] = str(meta_path)
    out['images_manifest_csv'] = str(manifest_csv)
    out['yolo_det_dir'] = str(yolo_dir)
    out['clip_embeddings_parquet'] = str(clip_parq)
    out['train_list'] = lists['train_list']
    out['val_list'] = lists['val_list']
    out['checksums_json'] = str(ds_dir / 'checksums.json')
    out['yolo_root'] = str(yolo_dir)
    return out


