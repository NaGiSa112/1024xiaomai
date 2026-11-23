# -*- coding: utf-8 -*-
"""在测试集/验证集上推理, 直接产出提交格式的 txt.

输出格式(与赛题要求一致, 每行一个目标, 空格分隔):
    <class_id> <x_center> <y_center> <width> <height> <confidence>
    归一化到 [0,1], class_id 从 0 开始(本赛题只有 0=wheat_ear)

用法:
  python scripts/predict.py --weights runs/detect/y11s_1024/weights/best.pt \
      --source data/dataset/images/test --imgsz 1024 --conf 0.001 --out preds/test_y11s_1024
  python scripts/predict.py ... --source data/dataset/images/val --out preds/val_y11s_1024
  python scripts/predict.py ... --tta                     # 多尺度/翻转 TTA, 涨点但慢 3 倍
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "_cfg" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cfg" / "mpl"))

import sandbox_compat  # noqa: F401

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="图片目录")
    p.add_argument("--out", required=True, help="txt 输出目录")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.001, help="mAP 评测建议 0.001(保留全部候选)")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU; 密集目标别太小")
    p.add_argument("--max-det", type=int, default=1000, help="单图最多检测数, 密集场景要放大")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--half", action="store_true", help="FP16 推理")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张(调试)")
    return p.parse_args()


def main():
    a = parse_args()
    src = Path(a.source)
    if not src.is_absolute():
        src = ROOT / src
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
    if a.limit:
        imgs = imgs[: a.limit]
    if not imgs:
        sys.exit(f"没有图片: {src}")

    model = YOLO(str(ROOT / a.weights if not os.path.isabs(a.weights) else a.weights))
    print(f"[predict] {len(imgs)} 张 | imgsz={a.imgsz} conf={a.conf} iou={a.iou} tta={a.tta} max_det={a.max_det}")

    t0 = time.time()
    n_box = 0
    empty = 0
    stats = []

    for i in range(0, len(imgs), a.batch):
        chunk = imgs[i : i + a.batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=a.imgsz,
            conf=a.conf,
            iou=a.iou,
            max_det=a.max_det,
            augment=a.tta,
            half=a.half,
            device=a.device,
            verbose=False,
            stream=False,
        )
        for p, r in zip(chunk, results):
            b = r.boxes
            names = r.names if hasattr(r, "names") else {0: "wheat_ear"}
            lines = []
            if b is not None and len(b):
                xywhn = b.xywhn.cpu().numpy().astype(np.float64)
                confs = b.conf.cpu().numpy().astype(np.float64)
                clss = b.cls.cpu().numpy().astype(int)
                for (xc, yc, w, h), c, k in zip(xywhn, confs, clss):
                    lines.append(f"{k} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {c:.6f}")
            body = "\n".join(lines) + ("\n" if lines else "")
            # newline="\n": 强制 LF, 避免 CRLF 在平台侧解析时混进最后一个字段
            with open(out / (p.stem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
            n_box += len(lines)
            empty += 0 if lines else 1
            stats.append(len(lines))
        done = min(i + a.batch, len(imgs))
        print(f"\r  {done}/{len(imgs)}  框={n_box}", end="", flush=True)

    dt = time.time() - t0
    stats = np.array(stats) if stats else np.array([0])
    summary = {
        "weights": a.weights,
        "source": str(src),
        "out": str(out),
        "images": len(imgs),
        "boxes": int(n_box),
        "boxes_per_image_mean": round(float(stats.mean()), 2),
        "boxes_per_image_max": int(stats.max()),
        "empty_images": int(empty),
        "imgsz": a.imgsz,
        "conf": a.conf,
        "iou": a.iou,
        "max_det": a.max_det,
        "tta": bool(a.tta),
        "seconds": round(dt, 1),
        "sec_per_image": round(dt / len(imgs), 3),
    }
    (out / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
