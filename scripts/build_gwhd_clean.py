# -*- coding: utf-8 -*-
"""构建"干净"外部训练集：GWHD 剔除所有与赛题数据同源的图（含 test）。

为什么要剔除 test 同源图：
  赛题规则写明「测试集仅可用于输出最终推理结果，不可以任何形式参与训练过程」。
  既然 GWHD 里存在与测试图同一张图的版本（及其标注），把它们放进训练集就等于
  用测试图训练 —— 必须排除，而且要在报告里写清楚这个筛查过程。

同时剔除与 train/val 同源的图：那是同一张图的重复（且标签口径不同），
留着只会制造标签冲突，不带来新信息。

输出：
  data_external/gwhd_clean_train.txt   —— 干净图片清单（绝对路径）
  data_external/gwhd_clean.yaml        —— 可直接喂 train.py
  data_external/gwhd_clean_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_overlap import dhash, popcount_matrix, thumb  # noqa: E402

DATA = ROOT / "data" / "dataset"
GWH = ROOT / "data_external" / "gwhd"
MAD_THR = 6.0


def main():
    comp = sorted((DATA / "images" / "train").glob("*.png")) + sorted((DATA / "images" / "val").glob("*.png")) + sorted(
        (DATA / "images" / "test").glob("*.png")
    )
    g_imgs = sorted((GWH / "images" / "train").glob("*.jpg")) + sorted((GWH / "images" / "val").glob("*.jpg"))
    print(f"赛题图 {len(comp)} 张 / GWHD {len(g_imgs)} 张")

    g_hash = np.array([dhash(p) for p in g_imgs], dtype=np.uint64)
    matched = {}          # gwhd 文件名 -> (赛题图名, MAD)
    for i, p in enumerate(comp):
        d = popcount_matrix(np.array([dhash(p)], dtype=np.uint64), g_hash).ravel()
        t = thumb(p)
        for j in np.argsort(d)[:3]:
            mad = float(np.abs(t - thumb(g_imgs[j])).mean())
            if mad < MAD_THR:
                name = g_imgs[j].name
                if name not in matched or mad < matched[name][1]:
                    matched[name] = (f"{p.parent.name}/{p.name}", mad)
        if (i + 1) % 200 == 0:
            print(f"  已比对 {i + 1}/{len(comp)}，命中 {len(matched)}")

    clean = [p for p in g_imgs if p.name not in matched]
    print(f"剔除同源图 {len(matched)} 张，剩余干净图 {len(clean)} 张")

    lst = ROOT / "data_external" / "gwhd_clean_train.txt"
    lst.write_text("\n".join(str(p) for p in clean) + "\n", encoding="utf-8")

    yml = ROOT / "data_external" / "gwhd_clean.yaml"
    yml.write_text(
        "# GWHD 干净子集：已剔除与赛题 train/val/test 同源（同一张图）的样本\n"
        "# 用途：预训练（不是最终评估）；val 指向赛题官方 val 用于观察收敛\n"
        f"path: {(ROOT / 'data' / 'dataset').as_posix()}\n"
        f"train: {lst.as_posix()}\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n  0: wheat_ear\n",
        encoding="utf-8",
    )

    summary = {
        "competition_images": len(comp),
        "gwhd_images": len(g_imgs),
        "excluded_same_image": len(matched),
        "excluded_breakdown": {
            s: sum(1 for v in matched.values() if v[0].startswith(s + "/")) for s in ("train", "val", "test")
        },
        "clean_images": len(clean),
        "mad_threshold": MAD_THR,
        "list": str(lst),
        "yaml": str(yml),
    }
    (ROOT / "data_external" / "gwhd_clean_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
