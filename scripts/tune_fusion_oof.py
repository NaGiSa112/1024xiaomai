# -*- coding: utf-8 -*-
"""用 5 折 OOF 预测来调集成参数（不碰测试集）。

思路：每个折模型只对它没见过的折外图片出预测，把 5 折拼起来就是 **900 张全都有"折外预测"**。
在这 900 张上调融合方式（NMS/WBF）、IoU 阈值、box_scale、成员子集，选出的配置再拿去跑 test。

目录约定（先用 predict.py --source-list 生成）：
  preds/oof/<tag>_<成员>_s<尺度>/fold<k>/*.txt
  例如 preds/oof/x900_1024_s1024/fold0/0001.txt

用法:
  # 1) 先出一份"所有成员 × 3 尺度"的 OOF 预测（见 run_big_ensemble.ps1）
  # 2) 搜参
  python scripts/tune_fusion_oof.py --root preds/oof --folds-dir data/folds --folds 0 1 2 3 4 ^
      --fuse nms wbf --iou 0.5 0.55 0.6 0.65 --box-scale 1.0 1.03
  # 3) 输出 preds/oof/tune_results.json（按 mAP50 排序）
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_submission import average_precision, iou_matrix, load_gt, load_pred  # noqa: E402


def gt_for(img: Path) -> Path:
    parts = list(img.parts)
    parts[len(parts) - 3] = "labels"
    return Path(*parts).with_suffix(".txt")


def iou_vec(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(ua > 0, inter / ua, 0.0)


def fuse(boxes, confs, kind, iou_thr, max_det, n_src, box_scale):
    if len(boxes) == 0:
        return boxes, confs
    if box_scale != 1.0:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = (boxes[:, 2] - boxes[:, 0]) * box_scale
        h = (boxes[:, 3] - boxes[:, 1]) * box_scale
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    if kind == "nms":
        keep = []
        order = np.argsort(-confs)
        suppressed = np.zeros(len(boxes), dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            ious = iou_vec(boxes[i], boxes)
            suppressed |= (ious > iou_thr) & (np.arange(len(boxes)) != i)
        keep = np.array(keep[:max_det])
        return boxes[keep], confs[keep]

    # WBF
    order = np.argsort(-confs)
    b, s = boxes[order], confs[order]
    K = len(b)
    cl_box = np.zeros((K, 4))
    cl_sum = np.zeros((K, 4))
    cl_w = np.zeros(K)
    cl_n = np.zeros(K, dtype=np.int32)
    cl_s = np.zeros(K)
    k = 0
    for i in range(K):
        if k:
            ious = iou_vec(b[i], cl_box[:k])
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                cl_sum[j] += b[i] * s[i]
                cl_w[j] += s[i]
                cl_n[j] += 1
                cl_s[j] += s[i]
                cl_box[j] = cl_sum[j] / cl_w[j]
                continue
        cl_box[k] = b[i]
        cl_sum[k] = b[i] * s[i]
        cl_w[k] = s[i]
        cl_n[k] = 1
        cl_s[k] = s[i]
        k += 1
    fb = cl_box[:k]
    fs = (cl_s[:k] / cl_n[:k]) * np.minimum(cl_n[:k], n_src) / max(n_src, 1)
    keep = np.argsort(-fs)[:max_det]
    return fb[keep], fs[keep]


def evaluate(members, imgs_by_fold, kind, iou_thr, max_det, box_scale, gt_cache, max_det_pred=1000):
    tp_all, conf_all, n_gt_all = [], [], 0
    n_src = len(members)
    for fold, imgs in imgs_by_fold.items():
        for img in imgs:
            stem, gt = img.stem, gt_cache[img]
            bl, cl = [], []
            for m in members:
                p = ROOT / "preds" / "oof" / m / f"fold{fold}" / f"{stem}.txt"
                if not p.exists():
                    continue
                b, c = load_pred(p)
                if len(b):
                    bl.append(b)
                    cl.append(c)
            n_gt_all += len(gt)
            if not bl:
                continue
            boxes = np.concatenate(bl)
            confs = np.concatenate(cl)
            boxes, confs = fuse(boxes, confs, kind, iou_thr, max_det, n_src, box_scale)
            if len(boxes) == 0:
                continue
            ious = iou_matrix(boxes, gt)
            order = np.argsort(-confs)
            matched = np.zeros(len(gt), dtype=bool)
            for idx in order:
                j = int(np.argmax(ious[idx])) if len(gt) else 0
                if len(gt) and ious[idx, j] >= 0.5 and not matched[j]:
                    matched[j] = True
                    tp_all.append(1)
                else:
                    tp_all.append(0)
                conf_all.append(confs[idx])
    if not tp_all:
        return 0.0
    tp = np.array(tp_all)
    conf = np.array(conf_all)
    order = np.argsort(-conf)
    cum_tp = np.cumsum(tp[order])
    cum_fp = np.cumsum(1 - tp[order])
    rec = cum_tp / max(n_gt_all, 1)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return average_precision(rec, prec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="preds/oof")
    ap.add_argument("--folds-dir", default="data/folds")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--members", nargs="+", default=None, help="成员目录名（默认自动发现）")
    ap.add_argument("--fuse", nargs="+", default=["nms", "wbf"])
    ap.add_argument("--iou", nargs="+", type=float, default=[0.55, 0.6])
    ap.add_argument("--box-scale", nargs="+", type=float, default=[1.0])
    ap.add_argument("--max-det", type=int, default=600)
    a = ap.parse_args()

    root = ROOT / a.root
    members = a.members or sorted([p.name for p in root.iterdir() if p.is_dir() and p.name != "tune_results.json"])
    if not members:
        raise SystemExit(f"{root} 下没有 OOF 预测目录")

    imgs_by_fold = {}
    gt_cache = {}
    for k in a.folds:
        lp = ROOT / a.folds_dir / f"fold{k}_val.txt"
        imgs = [Path(l.strip()) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        imgs_by_fold[k] = imgs
        for im in imgs:
            gt_cache[im] = load_gt(gt_for(im))

    print(f"[tune] 成员 {len(members)} 个 / OOF 图片 {sum(len(v) for v in imgs_by_fold.values())} 张", flush=True)
    results = []
    for kind, iou, bs in itertools.product(a.fuse, a.iou, a.box_scale):
        m = evaluate(members, imgs_by_fold, kind, iou, a.max_det, bs, gt_cache)
        results.append({"members": len(members), "fuse": kind, "iou": iou, "box_scale": bs, "mAP50": round(float(m), 5)})
        print(f"  {kind} iou={iou} box_scale={bs} -> mAP50 {m:.5f}", flush=True)

    results.sort(key=lambda r: r["mAP50"], reverse=True)
    (root / "tune_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n最佳组合：", json.dumps(results[0], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
