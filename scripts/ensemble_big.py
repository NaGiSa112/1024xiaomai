# -*- coding: utf-8 -*-
"""大集成引擎：任意多个成员 × 任意多个推理尺度 -> NMS / 加权 WBF -> 提交格式。

成员用 JSON 描述，支持每个成员单独的推理尺度与置信度权重：

  [
    {"path": "runs/detect/x900_1024/weights/best.pt",     "scales": [1024, 1280], "weight": 1.0},
    {"path": "runs/detect/l1280_900/weights/best.pt",     "scales": [1280, 1024], "weight": 1.0},
    {"path": "runs/detect/fold0_y11s_1024/weights/best.pt","scales": [896, 1024],  "weight": 0.8}
  ]

用法:
  python scripts/ensemble_big.py --members members.json --source data/dataset/images/test --out preds/big --fuse wbf
  python scripts/ensemble_big.py --members members.json --source data/dataset/images/test --out preds/big --fuse nms --tag big_nms

设计要点:
  - 逐成员逐尺度推理，全部映射到归一化 xyxy；
  - 每个成员的置信度乘 weight（WBF 里当成"票数权重"用，NMS 里只影响排序）；
  - 融合后按 conf 截断 max_det，坐标 clamp 到 [0,1]；
  - 输出与 predict.py 完全一致的 txt 格式（LF、UTF-8、6 字段），可直接 make_submission.py。
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


def iou_vec(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(ua > 0, inter / ua, 0.0)


def fuse_wbf(boxes, confs, iou_thr, max_det, n_sources):
    """加权 WBF：按 conf 降序聚类，融合框 = conf 加权平均，置信度 = 平均 conf × min(n,N)/N。"""
    if len(boxes) == 0:
        return boxes, confs
    order = np.argsort(-confs)
    boxes, confs = boxes[order], confs[order]
    b = boxes.astype(np.float64)
    s = confs.astype(np.float64)

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
    fs = (cl_s[:k] / cl_n[:k]) * np.minimum(cl_n[:k], n_sources) / max(n_sources, 1)
    keep = np.argsort(-fs)[:max_det]
    return fb[keep].astype(np.float32), fs[keep].astype(np.float32)


def fuse_nms(boxes, confs, iou_thr, max_det):
    if len(boxes) == 0:
        return boxes, confs
    from torchvision.ops import nms

    keep = nms(torch.from_numpy(boxes), torch.from_numpy(confs), float(iou_thr)).numpy()
    keep = keep[np.argsort(-confs[keep])][:max_det]
    return boxes[keep], confs[keep]


def scale_boxes(boxes, k):
    if k == 1.0 or len(boxes) == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]) * k
    h = (boxes[:, 3] - boxes[:, 1]) * k
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="成员 JSON（见文件头说明）")
    ap.add_argument("--source", default="data/dataset/images/test")
    ap.add_argument("--source-list", default=None, help="图片清单 txt（OOF 用）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuse", default="wbf", choices=["nms", "wbf"])
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--box-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tag", default=None, help="给了就顺便打 submit.zip")
    return ap.parse_args()


def run(args):
    members = json.loads(Path(args.members).read_text(encoding="utf-8-sig"))
    if args.source_list:
        lp = Path(args.source_list)
        if not lp.is_absolute():
            lp = ROOT / lp
        imgs = [Path(l.strip()) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        src = Path(args.source)
        if not src.is_absolute():
            src = ROOT / src
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    n_src = sum(len(m.get("scales", [1024])) for m in members)
    print(f"[big] {len(imgs)} 张 × {n_src} 个(成员,尺度) 组合，融合={args.fuse}", flush=True)

    agg = {p.stem: ([], []) for p in imgs}
    t0 = time.time()
    for mi, m in enumerate(members):
        mp = Path(m["path"])
        if not mp.is_absolute():
            mp = ROOT / mp
        if not mp.exists():
            print(f"  [warn] 成员缺失，跳过: {mp}", flush=True)
            continue
        w = float(m.get("weight", 1.0))
        model = YOLO(str(mp))
        for sz in m.get("scales", [1024]):
            t = time.time()
            for i in range(0, len(imgs), args.batch):
                chunk = imgs[i : i + args.batch]
                res = model.predict(
                    source=[str(p) for p in chunk],
                    imgsz=sz,
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
                    h, wd = r.orig_shape
                    xyxy = np.stack([xyxy[:, 0] / wd, xyxy[:, 1] / h, xyxy[:, 2] / wd, xyxy[:, 3] / h], axis=1)
                    agg[p.stem][0].append(xyxy)
                    agg[p.stem][1].append(conf * w)
            print(f"  [{mi+1}/{len(members)}] {mp.parent.parent.name}/{mp.name} @{sz}: {time.time()-t:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    n_box = 0
    counts = []
    for stem, (bl, cl) in agg.items():
        boxes = np.concatenate(bl) if bl else np.zeros((0, 4), np.float32)
        confs = np.concatenate(cl) if cl else np.zeros((0,), np.float32)
        boxes = scale_boxes(boxes, args.box_scale)
        if args.fuse == "wbf":
            boxes, confs = fuse_wbf(boxes, confs, args.wbf_iou, args.max_det, n_src)
        else:
            boxes, confs = fuse_nms(boxes, confs, args.nms_iou, args.max_det)

        lines = []
        for (x1, y1, x2, y2), s in zip(boxes, confs):
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            xc = min(max(float(xc), 0.0), 1.0)
            yc = min(max(float(yc), 0.0), 1.0)
            bw = min(max(float(bw), 1e-6), 1.0)
            bh = min(max(float(bh), 1e-6), 1.0)
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(s):.6f}")
        with open(out / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
        counts.append(len(lines))

    summary = {
        "members": len(members),
        "member_scale_pairs": n_src,
        "images": len(imgs),
        "boxes": n_box,
        "boxes_per_image_median": int(np.median(counts)) if counts else 0,
        "fuse": args.fuse,
        "iou": args.wbf_iou if args.fuse == "wbf" else args.nms_iou,
        "box_scale": args.box_scale,
        "seconds": round(time.time() - t0, 1),
        "out": str(out),
    }
    (out / "_big_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.tag:
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--pred", str(out), "--tag", args.tag],
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    args = main()
    run(args)
