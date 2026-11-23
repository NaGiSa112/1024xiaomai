# -*- coding: utf-8 -*-
"""Wheat-SOD 训练入口 (ultralytics YOLO).

示例:
  python train.py                                  # 默认 yolo11s @1024
  python train.py --model weights/yolo26s.pt --epochs 150 --name y26s_1024
  python train.py --resume                         # 续训最近一次
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "_cfg" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cfg" / "mpl"))
os.environ.setdefault("PYTHONUTF8", "1")

import sandbox_compat  # noqa: F401  必须在 ultralytics 之前

from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/yolo11s.pt", help="预训练权重")
    p.add_argument("--data", default="data/wheat.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024, help="麦穗很小, 建议 1024")
    p.add_argument("--batch", type=int, default=8, help="1024 输入下 4090 建议 8~16")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0, help="沙箱禁用多进程, 保持 0")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="none", choices=["none", "ram", "disk"])
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--close-mosaic", type=int, default=15)
    p.add_argument("--degrees", type=float, default=0.0, help="遥感/俯视图像不建议大角度旋转")
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.0)
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--no-val", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    name = a.name or f"{Path(a.model).stem}_{a.imgsz}"
    t0 = time.time()

    model = YOLO(str(ROOT / a.model) if not os.path.isabs(a.model) else a.model)

    if a.resume:
        # 续训: 指向最近一次 run 的 last.pt
        cands = sorted((ROOT / a.project).glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit("没有找到可续训的 last.pt")
        print(f"[resume] {cands[-1]}")
        model = YOLO(str(cands[-1]))
        a.resume = True

    print("=" * 68)
    print(f"model={a.model}  imgsz={a.imgsz}  batch={a.batch}  epochs={a.epochs}  name={name}")
    print("=" * 68)

    model.train(
        data=str(ROOT / a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=str(ROOT / a.project),
        name=name,
        exist_ok=a.exist_ok,
        resume=a.resume,
        seed=a.seed,
        patience=a.patience,
        cache=False if a.cache == "none" else a.cache,
        optimizer=a.optimizer,
        lr0=a.lr0,
        lrf=a.lrf,
        cos_lr=True,
        close_mosaic=a.close_mosaic,
        degrees=a.degrees,
        fliplr=a.fliplr,
        flipud=a.flipud,
        scale=a.scale,
        mosaic=a.mosaic,
        mixup=a.mixup,
        copy_paste=a.copy_paste,
        amp=True,
        val=not a.no_val,
        plots=True,
        verbose=True,
        deterministic=True,
        pretrained=True,
    )

    # 汇总
    run_dir = ROOT / a.project / name
    best = run_dir / "weights" / "best.pt"
    summary = {
        "model": a.model,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "epochs": a.epochs,
        "best": str(best),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    res_csv = run_dir / "results.csv"
    if res_csv.exists():
        import csv

        rows = list(csv.DictReader(open(res_csv, encoding="utf-8")))
        if rows:
            key = "metrics/mAP50-95(B)"
            rows_sorted = sorted(rows, key=lambda r: float(r.get(key, 0) or 0))
            best_row = rows_sorted[-1]
            summary["best_epoch"] = int(float(best_row.get("epoch", 0)))
            for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", key):
                summary[k] = round(float(best_row.get(k, 0) or 0), 5)

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 68)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"best.pt -> {best}")
    print("=" * 68)


if __name__ == "__main__":
    main()
