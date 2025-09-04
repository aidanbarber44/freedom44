import pandas as pd, numpy as np, torch, logging
from pathlib import Path
from typing import Optional


def set_torch_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_clip_embeddings(images_manifest_csv: str, out_parquet: str, model_name: str = "ViT-L-14", batch: int = 128, fp16: bool = True, seed: int = 0) -> None:
    import open_clip
    from PIL import Image

    set_torch_deterministic(seed)
    df = pd.read_csv(images_manifest_csv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s32b_b82k")
    model = model.to(device).eval()

    embs = []
    cols = [c for c in ['sha256','image_path','symbol','ts','crop_meta','split'] if c in df.columns]
    use_amp = (device == 'cuda') and fp16
    with torch.no_grad():
        for i in range(0, len(df), batch):
            paths = df['image_path'].iloc[i:i+batch].tolist()
            imgs = []
            for p in paths:
                try:
                    imgs.append(preprocess(Image.open(p).convert('RGB')).unsqueeze(0))
                except Exception:
                    # substitute a zero image on failure
                    imgs.append(torch.zeros(1, 3, 224, 224))
            batch_t = torch.cat(imgs, dim=0).to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    e = model.encode_image(batch_t)
            else:
                e = model.encode_image(batch_t)
            e = e.float().cpu().numpy()
            embs.append(e)
    E = np.vstack(embs) if len(embs) else np.zeros((0, 768), dtype=np.float32)
    emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
    out = pd.concat([df[cols].reset_index(drop=True), pd.DataFrame(E, columns=emb_cols)], axis=1)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)


