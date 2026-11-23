# -*- coding: utf-8 -*-
"""下载 GWHD（Global Wheat Head Detection）外部数据（HuggingFace 镜像, 已是 YOLO 格式）.

数据源: https://huggingface.co/datasets/dronefreak/GWHD
  data/images/{train,val}  +  data/labels/{train,val}
  标签: `<class> <xc> <yc> <w> <h>` 归一化, 单类 wheat_head —— 与本赛题格式完全一致

用法:
  python scripts/fetch_gwhd.py --list                 # 只统计数量和体积
  python scripts/fetch_gwhd.py --splits train val     # 下载
  python scripts/fetch_gwhd.py --splits train --max-per-split 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data_external" / "gwhd"
REPO = "dronefreak/GWHD"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
UA = {"User-Agent": "Mozilla/5.0"}

_lock = threading.Lock()
_done = [0]
_total = [0]


def _get(url: str, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def list_dir(path: str):
    """带游标的分页枚举, 返回 [(path, size)]."""
    out, url = [], f"{API}/{path}?limit=1000"
    while url:
        body, headers = _get(url)
        items = json.loads(body.decode("utf-8"))
        for it in items:
            if it.get("type") == "file":
                out.append((it["path"], int(it.get("size") or 0)))
        link = headers.get("Link") or headers.get("link") or ""
        url = None
        if 'rel="next"' in link:
            seg = [s for s in link.split(",") if 'rel="next"' in s][0]
            url = seg.split(";")[0].strip().strip("<>")
    return out


def fetch_one(path: str, size: int, retries=3):
    dst = DEST / path.replace("data/", "", 1)
    if dst.exists() and dst.stat().st_size == size:
        with _lock:
            _done[0] += 1
        return dst, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            body, _ = _get(f"{RAW}/{path}", timeout=180)
            if size and len(body) != size:
                raise IOError(f"size mismatch {len(body)}!={size}")
            dst.write_bytes(body)
            with _lock:
                _done[0] += 1
            return dst, len(body)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                return dst, -1
            time.sleep(1.5 * (i + 1))
    return dst, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--max-per-split", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    tasks = []
    for split in a.splits:
        imgs = list_dir(f"data/images/{split}")
        labs = list_dir(f"data/labels/{split}")
        lab_map = {Path(p).stem: (p, sz) for p, sz in labs}
        if a.max_per_split:
            imgs = sorted(imgs)[: a.max_per_split]
        n_missing = 0
        for p, sz in imgs:
            stem = Path(p).stem
            if stem not in lab_map:
                n_missing += 1
                continue
            tasks.append((p, sz))
            tasks.append(lab_map[stem])
        print(f"[{split}] imgs={len(imgs)} labels={len(labs)} 无标签跳过={n_missing}")
    total_mb = sum(sz for _, sz in tasks) / 1024**2
    print(f"合计 {len(tasks)} 个文件, {total_mb:.1f} MB")
    if a.list:
        return

    _total[0] = len(tasks)
    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, p, sz): p for p, sz in tasks}
        for f in as_completed(futs):
            dst, n = f.result()
            if n < 0:
                failed.append(str(dst))
            if _done[0] % 200 == 0 or _done[0] == _total[0]:
                el = time.time() - t0
                print(f"\r  {_done[0]}/{_total[0]}  {el:.0f}s", end="", flush=True)
    print()
    man = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "files": len(tasks) - len(failed),
        "failed": failed[:20],
        "mb": round(total_mb, 1),
        "seconds": round(time.time() - t0, 1),
    }
    (DEST / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
