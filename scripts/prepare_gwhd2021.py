# -*- coding: utf-8 -*-
"""处理 GWHD2021 全量包（gwhd_2021.zip）：解压 -> 转 YOLO -> 合规筛查 -> 去重 -> 生成 v2 干净集。

包结构：
  gwhd_2021/images/*.png            6515 张 1024×1024
  gwhd_2021/competition_{train,val,test}.csv
      image_name, BoxesString("x1 y1 x2 y2;..." 像素 xyxy), domain

合规与去重要求（两轮剔除）：
  1) **与赛题 1000 张同源的图全部剔除**（尤其 test 同源，规则禁止测试集参与训练）；
  2) 与已有外部数据（parquet 批次 3401 + dronefreak 追加 677）同源的图也剔除，避免重复计入。

用法:
  python scripts/prepare_gwhd2021.py --extract       # 解压 + 转标签 + 筛查 + 生成 v2
  python scripts/prepare_gwhd2021.py --skip-extract  # 只重跑筛查
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_overlap import dhash, popcount_matrix, thumb  # noqa: E402

ZIP = ROOT / "data_external" / "gwhd2021" / "gwhd_2021.zip"
DST = ROOT / "data_external" / "gwhd2021"
MAD_THR = 6.0


def extract_all():
    imgs_dir = DST / "images"
    lab_dir = DST / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.startswith("gwhd_2021/images/") and n.lower().endswith(".png")]
        print(f"[extract] 图片 {len(names)} 张", flush=True)
        for i, n in enumerate(names):
            out = imgs_dir / Path(n).name
            if not out.exists():
                out.write_bytes(z.read(n))
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)

        records = {}
        for split in ("train", "val", "test"):
            name = f"gwhd_2021/competition_{split}.csv"
            with z.open(name) as f:
                txt = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                for row in csv.DictReader(txt):
                    key = row["image_name"]
                    rec = records.setdefault(key, {"split": split, "domain": row.get("domain", ""), "boxes": []})
                    for b in (row.get("BoxesString") or "").split(";"):
                        parts = b.split()
                        if len(parts) != 4:
                            continue
                        try:
                            x1, y1, x2, y2 = [float(v) for v in parts]
                        except ValueError:
                            continue
                        rec["boxes"].append((x1, y1, x2, y2))

    n_box = 0
    for key, rec in records.items():
        stem = Path(key).stem
        lines = []
        for x1, y1, x2, y2 in rec["boxes"]:
            x1, x2 = max(0.0, min(x1, 1024)), max(0.0, min(x2, 1024))
            y1, y2 = max(0.0, min(y1, 1024)), max(0.0, min(y2, 1024))
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                continue
            lines.append(f"0 {(x1+x2)/2/1024:.6f} {(y1+y2)/2/1024:.6f} {w/1024:.6f} {h/1024:.6f}")
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_box += len(lines)
    print(f"[labels] {len(records)} 条记录 / {n_box} 个框", flush=True)

    meta = {"images": len(names), "records": len(records), "boxes": n_box}
    (DST / "extract_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def hashes_of(paths):
    out = []
    for p in paths:
        try:
            out.append(dhash(p))
        except Exception:  # noqa: BLE001
            out.append(np.uint64(0))
    return np.array(out, dtype=np.uint64)


def screen():
    comp = []
    for split in ("train", "val", "test"):
        comp += sorted((ROOT / "data" / "dataset" / "images" / split).glob("*.png"))
    old = []
    for sub in ("images/train", "images/val"):
        old += sorted((ROOT / "data_external" / "gwhd" / sub).glob("*.jpg"))
    old += sorted((ROOT / "data_external" / "gwhd_extra" / "images").rglob("*.jpg"))
    new = sorted((DST / "images").glob("*.png"))
    print(f"[screen] 赛题 {len(comp)} / 已有外部 {len(old)} / 新增 {len(new)}", flush=True)

    def mark(ref_paths, ref_label):
        ref_hash = hashes_of(ref_paths)
        # 参考图缩略图预缓存（每张只解码一次，否则 6515×3 次重复解码要十几分钟）
        ref_thumbs = []
        for i, p in enumerate(ref_paths):
            try:
                ref_thumbs.append(thumb(p))
            except Exception:  # noqa: BLE001
                ref_thumbs.append(np.zeros((128, 128, 3), dtype=np.int16))
            if (i + 1) % 1500 == 0:
                print(f"  [{ref_label}] 预缓存缩略图 {i+1}/{len(ref_paths)}", flush=True)

        hits = {}
        for i, p in enumerate(new):
            d = popcount_matrix(np.array([dhash(p)], dtype=np.uint64), ref_hash).ravel()
            t = thumb(p)
            for j in np.argsort(d)[:3]:
                mad = float(np.abs(t - ref_thumbs[j]).mean())
                if mad < MAD_THR:
                    hits[p.name] = (f"{ref_label}:{ref_paths[j].name}", round(mad, 2))
                    break
            if (i + 1) % 1500 == 0:
                print(f"  {ref_label}: {i+1}/{len(new)} 命中 {len(hits)}", flush=True)
        return hits

    hit_comp = mark(comp, "赛题")
    remain = [p for p in new if p.name not in hit_comp]
    hit_old = mark(remain, "旧外部") if remain else {}
    print(f"[screen] 与赛题同源 {len(hit_comp)} 张，与旧外部重复 {len(hit_old)} 张", flush=True)

    clean = [p for p in new if p.name not in hit_comp and p.name not in hit_old]
    lst = ROOT / "data_external" / "gwhd2021_clean_train.txt"
    lst.write_text("\n".join(str(p) for p in clean) + "\n", encoding="utf-8")

    # 与已有干净集合并
    old_clean = ROOT / "data_external" / "gwhd_clean_train.txt"
    old_lines = [l for l in old_clean.read_text(encoding="utf-8").splitlines() if l.strip()] if old_clean.exists() else []
    merged = old_lines + [str(p) for p in clean]
    mpath = ROOT / "data_external" / "gwhd_clean_v2_train.txt"
    mpath.write_text("\n".join(merged) + "\n", encoding="utf-8")
    yml = ROOT / "data_external" / "gwhd_clean_v2.yaml"
    yml.write_text(
        "# GWHD v2 干净集：GWHD2020(parquet) + dronefreak 追加 + GWHD2021 全量\n"
        "# 已剔除：与赛题 train/val/test 同源的图、以及与旧外部集重复的图\n"
        f"path: {(ROOT / 'data' / 'dataset').as_posix()}\n"
        f"train: {mpath.as_posix()}\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n  0: wheat_ear\n",
        encoding="utf-8",
    )

    per_split = {}
    for split in ("train", "val", "test"):
        per_split[split] = sum(1 for v in hit_comp.values() if v[0].startswith(f"赛题:") and False)
    summary = {
        "new_images": len(new),
        "excluded_same_as_competition": len(hit_comp),
        "excluded_dup_of_old_external": len(hit_old),
        "clean_new": len(clean),
        "old_clean": len(old_lines),
        "merged_total": len(merged),
        "list": str(mpath),
        "yaml": str(yml),
        "examples_competition_hits": list(hit_comp.items())[:5],
    }
    (ROOT / "data_external" / "gwhd2021_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()

    if not a.skip_extract:
        if not ZIP.exists():
            raise SystemExit(f"缺 zip: {ZIP}")
        if not (DST / "images").exists() or len(list((DST / "images").glob("*.png"))) < 100:
            extract_all()
        else:
            print("[skip] 图片已解压")
    screen()


if __name__ == "__main__":
    main()
