import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip


def set_torch_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False


def safe_encode(model, preprocess, paths: List[str], device: str, batch: int = 128, fp16: bool = True) -> np.ndarray:
    """OOM-safe encoder that backs off batch size on CUDA."""
    out = []
    i, n = 0, len(paths)
    cur = batch
    while i < n:
        try:
            chunk = paths[i:i + cur]
            imgs = []
            for p in chunk:
                try:
                    imgs.append(preprocess(Image.open(p).convert("RGB")).unsqueeze(0))
                except Exception:
                    # keep shape consistent for bad images
                    imgs.append(torch.zeros(1, 3, 224, 224))
            t = torch.cat(imgs, 0).to(device, non_blocking=True)
            with torch.no_grad():
                if fp16 and device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        e = model.encode_image(t)
                else:
                    e = model.encode_image(t)
            out.append(e.float().cpu().numpy())
            i += cur
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur > 8:
                torch.cuda.empty_cache()
                cur = max(8, cur // 2)
            else:
                raise
    return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def build_clip_embeddings(
    images_manifest_csv: str,
    out_parquet: str,
    model_name: str = "ViT-L-14",
    pretrained: str = "laion2b_s32b_b82k",
    batch: int = 128,
    fp16: bool = True,
    seed: int = 0,
) -> None:
    """
    Build CLIP embeddings for images listed in images_manifest_csv.
    Assumes a column 'image_path'. Will preserve any of: symbol, ts, ts_iso, split, view, sha256.
    """
    set_torch_deterministic(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(images_manifest_csv)
    assert "image_path" in df.columns, f"image_path column missing in {images_manifest_csv}"
    keep_cols = [c for c in ["image_path", "symbol", "ts", "ts_iso", "split", "view", "sha256"] if c in df.columns]

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)

    paths = df["image_path"].tolist()
    E = safe_encode(model, preprocess, paths, device=device, batch=batch, fp16=fp16)

    emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
    out = pd.concat(
        [df[keep_cols].reset_index(drop=True), pd.DataFrame(E, columns=emb_cols)],
        axis=1
    )
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)



