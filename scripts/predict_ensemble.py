# -*- coding: utf-8 -*-
"""多模型 x 多尺度 集成推理 -> 提交格式 txt.

队友已验证: 同族集成会饱和、翻转 TTA 有害、多尺度有效、简化 WBF 无增益.
所以这里的策略是: 跨架构 + 多尺度 + NMS(可选标准 WBF) + 可选框尺寸微调.

用法:
  # 单模型多尺度
  python scripts/predict_ensemble.py --weights runs/detect/A/weights/best.pt \
      --source data/dataset/images/test --out preds/ens_A --imgsz 896 1024 1280

  # 跨架构 + 多尺度 + NMS
  python scripts/predict_ensemble.py \
      --weights runs/detect/A/weights/best.pt runs/detect/B/weights/best.pt \
      --source data/dataset/images/test --out preds/ens_AB --imgsz 1024 1280 --fuse nms --nms-iou 0.6

  # 框尺寸微调(±3%)试探 GT 口径
  python scripts/predict_ensemble.py ... --box-scale 1.03
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

import sandbox_compat  # noqa: F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", nargs="+", type=int, default=[1024])
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--fuse", default="nms", choices=["nms", "wbf"])
    p.add_argument("--nms-iou", type=float, default=0.6)
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--box-scale", type=float, default=1.0, help="在中心点缩放框宽高, 用于对齐 GT 口径")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def collect(weights, imgsz, imgs, args):
    """返回 {stem: (xyxy_norm[N,4], conf[N])}, 全部在归一化坐标系."""
    model = YOLO(str(weights))
    acc: dict[str, list] = {p.stem: [] for p in imgs}
    for i in range(0, len(imgs), args.batch):
        chunk = imgs[i : i + args.batch]
        res = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            stream=False,
        )
        for p, r in zip(chunk, res):
            b = r.boxes
            if b is None or not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            conf = b.conf.cpu().numpy().astype(np.float32)
            # 像素 -> 归一化
            h, w = r.orig_shape
            xyxy = np.stack([xyxy[:, 0] / w, xyxy[:, 1] / h, xyxy[:, 2] / w, xyxy[:, 3] / h], axis=1)
            acc[p.stem].append((xyxy, conf))
    return acc


def scale_boxes(xyxy, k):
    if k == 1.0 or len(xyxy) == 0:
        return xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = (xyxy[:, 2] - xyxy[:, 0]) * k
    h = (xyxy[:, 3] - xyxy[:, 1]) * k
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms_fuse(xyxy, conf, iou_thr, max_det):
    if len(xyxy) == 0:
        return xyxy, conf
    from torchvision.ops import nms

    keep = nms(
        torch.from_numpy(xyxy),
        torch.from_numpy(conf),
        float(iou_thr),
    ).numpy()
    keep = keep[np.argsort(-conf[keep])][:max_det]
    return xyxy[keep], conf[keep]


def wbf_fuse(xyxy, conf, iou_thr, max_det, n_models):
    """标准 WBF: 按 conf 排序聚类(与已有簇做 IoU), 融合框 = conf 加权平均.

    注意: 簇匹配必须向量化 —— 每框与"所有现存簇"一次算 IoU.
    逐簇调用 numpy 的写法在 ~1400 框/图时会慢到 30 s/图.
    """
    if len(xyxy) == 0:
        return xyxy, conf
    order = np.argsort(-conf)
    boxes = xyxy[order]
    scores = conf[order]

    cl_box = np.zeros((len(boxes), 4), dtype=np.float64)   # 簇当前融合框(增量)
    cl_sum = np.zeros((len(boxes), 4), dtype=np.float64)   # Σ conf*box
    cl_w = np.zeros(len(boxes), dtype=np.float64)          # Σ conf
    cl_n = np.zeros(len(boxes), dtype=np.int32)            # 成员数
    cl_s = np.zeros(len(boxes), dtype=np.float64)          # Σ conf (用于均值)
    k = 0

    for box, s in zip(boxes.astype(np.float64), scores.astype(np.float64)):
        if k:
            ious = _iou(box[None], cl_box[:k])[0]      # (k,)
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                cl_sum[j] += box * s
                cl_w[j] += s
                cl_n[j] += 1
                cl_s[j] += s
                cl_box[j] = cl_sum[j] / cl_w[j]
                continue
        cl_box[k] = box
        cl_sum[k] = box * s
        cl_w[k] = s
        cl_n[k] = 1
        cl_s[k] = s
        k += 1

    fused_b = cl_box[:k]
    fused_s = (cl_s[:k] / cl_n[:k]) * np.minimum(cl_n[:k], n_models) / max(n_models, 1)
    keep = np.argsort(-fused_s)[:max_det]
    return fused_b[keep].astype(np.float32), fused_s[keep].astype(np.float32)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / ua, 0.0)


def main():
    args = parse_args()
    src = Path(args.source) if Path(args.source).is_absolute() else ROOT / args.source
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        sys.exit(f"没有图片: {src}")

    n_src = len(args.weights) * len(args.imgsz)
    print(f"[ens] {len(imgs)} 张 x {n_src} 个(模型,尺度) 组合 | fuse={args.fuse}")

    agg: dict[str, tuple[list, list]] = {p.stem: ([], []) for p in imgs}
    t0 = time.time()
    for w in args.weights:
        wp = Path(w) if Path(w).is_absolute() else ROOT / w
        for sz in args.imgsz:
            t = time.time()
            acc = collect(wp, sz, imgs, args)
            for stem, pairs in acc.items():
                for xyxy, conf in pairs:
                    agg[stem][0].append(xyxy)
                    agg[stem][1].append(conf)
            print(f"  {wp.parent.parent.name}/{wp.name} @{sz}: {time.time() - t:.1f}s")

    n_box = 0
    for stem, (bl, cl) in agg.items():
        xyxy = np.concatenate(bl) if bl else np.zeros((0, 4), np.float32)
        conf = np.concatenate(cl) if cl else np.zeros((0,), np.float32)
        xyxy = scale_boxes(xyxy, args.box_scale)
        if args.fuse == "nms":
            xyxy, conf = nms_fuse(xyxy, conf, args.nms_iou, args.max_det)
        else:
            xyxy, conf = wbf_fuse(xyxy, conf, args.wbf_iou, args.max_det, n_src)

        lines = []
        for (x1, y1, x2, y2), s in zip(xyxy, conf):
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
            bw, bh = min(max(bw, 1e-6), 1.0), min(max(bh, 1e-6), 1.0)
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(s):.6f}")
        body = "\n".join(lines) + ("\n" if lines else "")
        with open(out / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        n_box += len(lines)

    summary = {
        "weights": [str(w) for w in args.weights],
        "imgsz": args.imgsz,
        "fuse": args.fuse,
        "nms_iou": args.nms_iou,
        "conf_thr": args.conf,
        "box_scale": args.box_scale,
        "images": len(imgs),
        "boxes": n_box,
        "boxes_per_image": round(n_box / len(imgs), 1),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
