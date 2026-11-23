# -*- coding: utf-8 -*-
"""环境 + 数据集自检.  用法: python scripts/check_env.py"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "_cfg" / "ultralytics"))

import sandbox_compat  # noqa: F401  (must precede ultralytics)

OK, BAD = "[OK]", "[!!]"
problems = []


def head(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ------------------------------------------------------------------ 1. runtime
head("1. Python 运行时")
print(f"python  : {sys.version.split()[0]}  ({sys.executable})")
print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

import numpy as np
import cv2
import torch
import torchvision
import ultralytics
import yaml

print(f"numpy   : {np.__version__}")
print(f"opencv  : {cv2.__version__}")
print(f"torch   : {torch.__version__}   (cuda build {torch.version.cuda})")
print(f"torchvision: {torchvision.__version__}")
print(f"ultralytics: {ultralytics.__version__}")

head("2. GPU / CUDA")
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    print(f"{OK} cuda available, devices={torch.cuda.device_count()}")
    print(f"     {dev.name}  {dev.total_memory / 1024**3:.1f} GB  cc{dev.major}.{dev.minor}")
    a = torch.randn(2048, 2048, device="cuda")
    t = time.time()
    for _ in range(10):
        a = a @ a
    torch.cuda.synchronize()
    print(f"     matmul 2048^3 x10: {time.time() - t:.3f}s")
    del a
    torch.cuda.empty_cache()
else:
    problems.append("CUDA 不可用")
    print(f"{BAD} CUDA 不可用")

head("3. torchvision NMS / 推理链路")
try:
    from torchvision.ops import nms

    b = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
    s = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    kept = nms(b, s, 0.5)
    print(f"{OK} torchvision.ops.nms on cuda -> keep {kept.tolist()}")
except Exception as e:  # noqa: BLE001
    problems.append(f"nms 失败: {e}")
    print(f"{BAD} {e}")

# ------------------------------------------------------------------ 4. dataset
head("4. 数据集")
data_yaml = ROOT / "data" / "wheat.yaml"
cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
root = Path(cfg["path"])
print(f"yaml    : {data_yaml}")
print(f"root    : {root}  exists={root.exists()}")
print(f"classes : {cfg['nc']} -> {cfg['names']}")

from PIL import Image

for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
    img_dir = root / cfg[key].replace("images", "images")
    lab_dir = root / cfg[key].replace("images", "labels")
    imgs = sorted(img_dir.glob("*"))
    labs = sorted(lab_dir.glob("*.txt")) if lab_dir.exists() else []
    n_box = 0
    sizes = {}
    empty = 0
    bad = 0
    for p in imgs:
        with Image.open(p) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
        lab = lab_dir / (p.stem + ".txt")
        if not lab.exists():
            bad += 1
            continue
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        n_box += len(lines)
    print(
        f"{split:<5} imgs={len(imgs):<4} labels={len(labs):<4} boxes={n_box:<6} "
        f"avg={n_box / max(len(imgs), 1):.1f}  empty={empty} missing={bad}  sizes={sizes}"
    )
    if bad and split != "test":  # test 集本来就没有标注
        problems.append(f"{split}: {bad} 张图缺标注")

# 目标尺寸分布(相对边长) —— 判断是否属于小目标场景
head("5. 目标尺度分布 (train, 相对图像边长)")
ratios_w, ratios_h, areas = [], [], []
lab_dir = root / "labels" / "train"
for lab in lab_dir.glob("*.txt"):
    for line in lab.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        w, h = float(p[3]), float(p[4])
        ratios_w.append(w)
        ratios_h.append(h)
        areas.append(w * h)
if areas:
    rw = np.array(ratios_w)
    areas = np.array(areas)
    print(f"boxes={len(areas)}  w mean={rw.mean():.4f}  median={np.median(rw):.4f}  p95={np.percentile(rw, 95):.4f}")
    print(f"sqrt(area) median={np.sqrt(np.median(areas)):.4f}  -> 1024px 图上约 {np.sqrt(np.median(areas)) * 1024:.0f} px")
    tiny = (areas < (32 / 1024) ** 2).mean() * 100
    print(f"小于 32x32 的目标占比: {tiny:.1f}%")

# ------------------------------------------------------------------ 6. 训练栈
head("6. ultralytics 训练栈冒烟测试")
try:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / "yolo11n.pt"))
    t = time.time()
    res = model.predict(str(next((root / "images" / "val").glob("*.png"))), imgsz=640, device=0, verbose=False)
    dt = time.time() - t
    n = len(res[0].boxes) if res and res[0].boxes is not None else 0
    print(f"{OK} 单图推理 {dt * 1000:.0f} ms, 检出 {n} 个框 (COCO 预训练权重, 仅验证链路)")
except Exception as e:  # noqa: BLE001
    problems.append(f"ultralytics 推理失败: {e}")
    print(f"{BAD} {e}")

head("结论")
if problems:
    print(f"{BAD} 发现 {len(problems)} 个问题:")
    for p in problems:
        print("   -", p)
    sys.exit(1)
print(f"{OK} 环境自检全部通过, 可以开始训练")
