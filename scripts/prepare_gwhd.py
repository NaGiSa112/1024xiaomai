# -*- coding: utf-8 -*-
"""把 GWHD parquet（HF 镜像 yinsights8/global-wheat-detection-hf）转成 YOLO 数据集.

parquet 列: path(str), image(bytes,JPEG), width, height,
           boxes(list[[x1,y1,x2,y2,area]] 像素坐标), labels(list[int]), area
输出: data_external/gwhd/{images,labels}/{train,val}/  + data_external/gwhd.yaml

用法:
  python scripts/prepare_gwhd.py            # 全量转换
  python scripts/prepare_gwhd.py --limit 500
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_external" / "gwhd_parquet"
DST = ROOT / "data_external" / "gwhd"

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()

    files = sorted(SRC.glob("*.parquet"))
    if not files:
        sys.exit(f"没有 parquet: {SRC}")

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = 0
    n_box = 0
    n_empty = 0
    idx = 0
    stats = []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=200, columns=["image_id", "image", "width", "height", "boxes", "labels"]):
            for r in batch.to_pylist():
                if a.limit and idx >= a.limit:
                    break
                stem = r["image_id"]
                w, h = float(r["width"]), float(r["height"])
                boxes = r["boxes"] or []
                labels = r["labels"] or []
                # 按 10% 固定分配到 val（用索引取模，保证可复现）
                split = "val" if (idx % 100) < int(a.val_frac * 100) else "train"
                (DST / "images" / split / f"{stem}.jpg").write_bytes(r["image"]["bytes"])
                lines = []
                for b, lab in zip(boxes, labels):
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue
                    lines.append(f"{int(lab)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                with open(DST / "labels" / split / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                if not lines:
                    n_empty += 1
                n_box += len(lines)
                rows += 1
                idx += 1
                stats.append(len(lines))
            if a.limit and idx >= a.limit:
                break
        print(f"  {f.name} 累计 {rows} 张")

    yaml_path = ROOT / "data_external" / "gwhd.yaml"
    yaml_path.write_text(
        "# GWHD 外部数据（用于预训练 / 共训, 不参与本地指标）\n"
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n  0: wheat_ear\n",
        encoding="utf-8",
    )
    stats = np.array(stats)
    out = {
        "images": rows,
        "boxes": n_box,
        "boxes_per_image": round(float(stats.mean()), 2),
        "empty_labels": n_empty,
        "yaml": str(yaml_path),
    }
    (DST / "prepare_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
