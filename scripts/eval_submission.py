# -*- coding: utf-8 -*-
"""本地 mAP@0.5 校验: 用提交格式的 txt 对比 GT, 独立于 ultralytics 再算一遍.

这样能在提交前发现坐标转换 / 归一化 / class_id / 排序 等相关 bug,
而不是等平台跑完才知道格式错了.

用法:
  python scripts/eval_submission.py --pred preds/val_y11s_1024 \
      --gt data/dataset/labels/val --images data/dataset/images/val
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np  # noqa: E402


def load_gt(p: Path):
    boxes = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.split()
            if len(s) < 5:
                continue
            _, xc, yc, w, h = s[:5]
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def load_pred(p: Path):
    boxes, confs = [], []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.split()
            if len(s) < 6:
                continue
            _, xc, yc, w, h, c = s[:6]
            xc, yc, w, h, c = float(xc), float(yc), float(w), float(h), float(c)
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            confs.append(c)
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(confs, dtype=np.float64)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None], b[:, 1][None], b[:, 2][None], b[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / ua, 0.0)


def average_precision(rec, prec):
    """COCO 风格 101 点插值 AP."""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True, help="对应图片目录(用来确定要评估哪些 stem)")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    pred_dir = Path(a.pred) if Path(a.pred).is_absolute() else ROOT / a.pred
    gt_dir = Path(a.gt) if Path(a.gt).is_absolute() else ROOT / a.gt
    img_dir = Path(a.images) if Path(a.images).is_absolute() else ROOT / a.images
    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    n_gt = 0
    tp_flags, confs = [], []
    for stem in stems:
        g = load_gt(gt_dir / f"{stem}.txt")
        p, c = load_pred(pred_dir / f"{stem}.txt")
        n_gt += len(g)
        if len(p) == 0:
            continue
        ious = iou_matrix(p, g)
        order = np.argsort(-c)
        matched = np.zeros(len(g), dtype=bool)
        for idx in order:
            j = int(np.argmax(ious[idx]))
            if len(g) and ious[idx, j] >= a.iou and not matched[j]:
                matched[j] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)
            confs.append(c[idx])

    if not tp_flags:
        sys.exit("没有可评估的预测框")

    tp_flags = np.array(tp_flags)
    confs = np.array(confs)
    order = np.argsort(-confs)
    tp_flags = tp_flags[order]
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - tp_flags)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1e-12)
    mAP = average_precision(rec, prec)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1))

    out = {
        "images": len(stems),
        "gt_boxes": int(n_gt),
        "pred_boxes": int(len(confs)),
        "mAP@0.5": round(mAP, 5),
        "best_f1": round(float(f1[best]), 4),
        "best_f1_conf": round(float(confs[order][best]), 4),
        "precision_at_best": round(float(prec[best]), 4),
        "recall_at_best": round(float(rec[best]), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (pred_dir / "_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
