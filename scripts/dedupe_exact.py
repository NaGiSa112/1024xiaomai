# -*- coding: utf-8 -*-
"""精确去重：用"解码后像素的 md5"判断两张图是否同一张（重编码会变字节、但像素一致）。

背景：`prepare_gwhd2021.py` 的第二道筛查用「dHash + 像素 MAD<6」，在 6515×3373 规模上
假阳性过高（把 5515 张全判成重复）。这里改用**精确定义**：解码成 numpy 后取 md5。

输出:
  data_external/gwhd2021_clean_train.txt   仅含"与赛题 1000 张、与旧外部集都不重复"的 GWHD2021 图
  data_external/gwhd_clean_v2_train.txt    旧干净集 + 上述新增
  data_external/gwhd2021_dedupe_summary.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def fp(path: Path) -> str:
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.md5(a.tobytes()).hexdigest()


def collect(paths, label):
    out = {}
    for i, p in enumerate(paths):
        try:
            out[fp(p)] = p.name
        except Exception:  # noqa: BLE001
            continue
        if (i + 1) % 2000 == 0:
            print(f"  [{label}] {i+1}/{len(paths)}", flush=True)
    print(f"[{label}] 指纹 {len(out)} 个（来自 {len(paths)} 张）", flush=True)
    return out


def main():
    comp = []
    for split in ("train", "val", "test"):
        comp += sorted((ROOT / "data" / "dataset" / "images" / split).glob("*.png"))
    old = sorted((ROOT / "data_external" / "gwhd" / "images" / "train").glob("*.jpg"))
    old += sorted((ROOT / "data_external" / "gwhd" / "images" / "val").glob("*.jpg"))
    old += sorted((ROOT / "data_external" / "gwhd_extra" / "images").rglob("*"))
    new = sorted((ROOT / "data_external" / "gwhd2021" / "images").glob("*.png"))
    print(f"赛题 {len(comp)} / 旧外部 {len(old)} / GWHD2021 {len(new)}", flush=True)

    f_comp = collect(comp, "赛题")
    f_old = collect(old, "旧外部")
    f_new = {}
    for i, p in enumerate(new):
        try:
            f_new[p] = fp(p)
        except Exception:  # noqa: BLE001
            continue
        if (i + 1) % 2000 == 0:
            print(f"  [GWHD2021] {i+1}/{len(new)}", flush=True)

    known = set(f_comp) | set(f_old)
    clean = [p for p, h in f_new.items() if h not in known]
    n_in_comp = sum(1 for h in f_new.values() if h in f_comp)
    n_in_old = sum(1 for h in f_new.values() if h in f_old)
    print(f"[结果] GWHD2021 {len(f_new)} 张：命中赛题 {n_in_comp}，命中旧外部 {n_in_old}，**全新 {len(clean)}**", flush=True)

    lst = ROOT / "data_external" / "gwhd2021_clean_train.txt"
    lst.write_text("\n".join(str(p) for p in clean) + "\n", encoding="utf-8")

    old_clean = ROOT / "data_external" / "gwhd_clean_train.txt"
    old_lines = [l for l in old_clean.read_text(encoding="utf-8").splitlines() if l.strip()]
    merged = old_lines + [str(p) for p in clean]
    mpath = ROOT / "data_external" / "gwhd_clean_v2_train.txt"
    mpath.write_text("\n".join(merged) + "\n", encoding="utf-8")
    yml = ROOT / "data_external" / "gwhd_clean_v2.yaml"
    yml.write_text(
        "# GWHD v2 干净集：GWHD2020(parquet) + dronefreak 追加 + GWHD2021 全新图\n"
        "# 已按「解码后像素 md5」精确剔除：与赛题 1000 张重复的、与旧外部集重复的\n"
        f"path: {(ROOT / 'data' / 'dataset').as_posix()}\n"
        f"train: {mpath.as_posix()}\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n  0: wheat_ear\n",
        encoding="utf-8",
    )
    summary = {
        "competition_images": len(f_comp),
        "old_external_images": len(f_old),
        "gwhd2021_images": len(f_new),
        "gwhd2021_matching_competition": n_in_comp,
        "gwhd2021_matching_old_external": n_in_old,
        "gwhd2021_brand_new": len(clean),
        "v2_total": len(merged),
        "list_new": str(lst),
        "list_v2": str(mpath),
        "yaml_v2": str(yml),
    }
    (ROOT / "data_external" / "gwhd2021_dedupe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
