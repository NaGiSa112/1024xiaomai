# -*- coding: utf-8 -*-
"""可视化标注, 用于确认标签与图像对齐.

用法: python scripts/viz_labels.py [张数] [split]

注意: Windows 下 cv2.imread/imwrite 不支持中文路径, 这里统一走
      np.fromfile + cv2.imdecode / cv2.imencode + tofile.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
split = sys.argv[2] if len(sys.argv) > 2 else "val"
img_dir = ROOT / "data" / "dataset" / "images" / split
lab_dir = ROOT / "data" / "dataset" / "labels" / split
out_dir = ROOT / "docs" / "figs"
out_dir.mkdir(parents=True, exist_ok=True)


def imread_u(path: Path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


for img_path in sorted(img_dir.glob("*.png"))[:N]:
    lab = lab_dir / (img_path.stem + ".txt")
    im = imread_u(img_path)
    if im is None:
        print(f"读取失败: {img_path}")
        continue
    h, w = im.shape[:2]
    n = 0
    if lab.exists():
        for line in lab.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, xc, yc, bw, bh = p
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n += 1
    out = out_dir / f"gt_{split}_{img_path.stem}.jpg"
    imwrite_u(out, im)
    print(f"{out}  boxes={n}")
