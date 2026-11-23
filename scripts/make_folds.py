# -*- coding: utf-8 -*-
"""把 900 张（train 800 + val 100）做 5 折交叉验证划分，按"框密度"分层。

为什么按密度分层：这张数据集里单图框数 1~190，密度是最大的难度因子；
随机切分会让某折集中出现"稠密图"，OOF 指标噪声变大。

输出:
  data/folds/fold{0..4}_train.txt   每行一个图片绝对路径
  data/folds/fold{0..4}_val.txt
  data/folds/fold{0..4}.yaml        可直接喂给 train.py --data
  data/folds/folds_meta.json        每折样本数/框数/密度分布

用法: python scripts/make_folds.py [--folds 5] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
OUT = ROOT / "data" / "folds"


def collect():
    items = []
    for split in ("train", "val"):
        for img in sorted((DATA / "images" / split).glob("*.png")):
            lab = DATA / "labels" / split / f"{img.stem}.txt"
            n = len([l for l in lab.read_text().splitlines() if l.strip()]) if lab.exists() else 0
            items.append({"img": img, "split": split, "boxes": n})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = collect()
    if not items:
        raise SystemExit("没有找到数据")
    OUT.mkdir(parents=True, exist_ok=True)

    # 按密度分 5 个层（分位数），每层内部打乱后轮流分配到各折
    dens = np.array([it["boxes"] for it in items])
    qs = np.quantile(dens, np.linspace(0, 1, 6))
    qs[-1] += 1
    for i, it in enumerate(items):
        it["stratum"] = int(np.searchsorted(qs, it["boxes"], side="right") - 1)
        it["stratum"] = min(max(it["stratum"], 0), 4)

    rng = random.Random(a.seed)
    buckets = {k: [] for k in range(5)}
    for s in range(5):
        group = [it for it in items if it["stratum"] == s]
        rng.shuffle(group)
        for j, it in enumerate(group):
            buckets[j % a.folds].append(it)

    meta = {"folds": a.folds, "seed": a.seed, "total_images": len(items), "total_boxes": int(dens.sum()), "splits": []}
    for k in range(a.folds):
        val = sorted(buckets[k], key=lambda x: x["img"].name)
        train = [it for j in range(a.folds) if j != k for it in buckets[j]]
        (OUT / f"fold{k}_train.txt").write_text("\n".join(str(it["img"]) for it in train) + "\n", encoding="utf-8")
        (OUT / f"fold{k}_val.txt").write_text("\n".join(str(it["img"]) for it in val) + "\n", encoding="utf-8")
        yml = {
            "path": str(DATA),
            "train": str(OUT / f"fold{k}_train.txt"),
            "val": str(OUT / f"fold{k}_val.txt"),
            "nc": 1,
            "names": {0: "wheat_ear"},
        }
        (OUT / f"fold{k}.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
        vb = sum(it["boxes"] for it in val)
        tb = sum(it["boxes"] for it in train)
        meta["splits"].append(
            {
                "fold": k,
                "train_imgs": len(train),
                "val_imgs": len(val),
                "train_boxes": tb,
                "val_boxes": vb,
                "val_density_mean": round(float(np.mean([it["boxes"] for it in val])), 2),
                "val_from_original_val": sum(1 for it in val if it["split"] == "val"),
            }
        )

    (OUT / "folds_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
