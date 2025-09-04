import os, logging
from typing import Optional, List


def train_yolo(data_yaml: str, model_pref: str = "yolov8m.pt", imgsz: int = 896, epochs: int = 80) -> int:
    from ultralytics import YOLO
    tried = []
    for try_imgsz in [imgsz, 768, 640]:
        try:
            model = YOLO(model_pref)
            model.train(
                data=data_yaml, epochs=epochs, imgsz=try_imgsz, workers=2, device=0,
                mosaic=1.0, erasing=0.4, agnostic_nms=False, iou=0.65, max_det=1200,
                batch=-1, seed=0, deterministic=True, save=True, name="train_ict"
            )
            return try_imgsz
        except RuntimeError as e:
            tried.append((try_imgsz, str(e)))
            if "out of memory" in str(e).lower():
                continue
            raise
    logging.error(f"YOLO training failed, attempts={tried}")
    raise RuntimeError("YOLO training failed even at imgsz=640")


def write_data_yaml_from_lists(train_list: str, val_list: str, classes: List[str], out_yaml: str):
    content = [
        f"train: {train_list}",
        f"val: {val_list}",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(classes)]
    with open(out_yaml, "w") as f:
        f.write("\n".join(content) + "\n")
    return out_yaml


