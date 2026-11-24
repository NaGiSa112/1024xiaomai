# -*- coding: utf-8 -*-
"""追加下载 GWHD 之外部图片（同一 HF 镜像里我们还没拉的那部分），支持断点与高并发。

背景：
  - 当前 data_external/gwhd 是通过 parquet 镜像下的 3373 张；
  - 同一 HF 仓库 dronefreak/GWHD 实际有 train 3658 + val 1477 = 5135 张（YOLO 格式），
    还有约 1762 张我们没下；
  - 之前逐文件下载只有 0.3 文件/秒（16 线程），所以换 parquet；
    现在用 32 线程 + 断点续传，值得再试。

用法：
  python scripts/fetch_gwhd_more.py --list                 # 看还差多少
  python scripts/fetch_gwhd_more.py --workers 32           # 下载缺失的
  python scripts/fetch_gwhd_more.py --workers 48 --splits train val

下载后必须重跑 scripts/build_gwhd_clean.py 的筛查（排除与赛题 test 同源的图）。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data_external" / "gwhd_extra"
REPO = "dronefreak/GWHD"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
UA = {"User-Agent": "Mozilla/5.0"}
HAVE = ROOT / "data_external" / "gwhd"          # 已有图片（parquet 那批）


def _get(url: str, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def list_dir(path: str):
    out, url = [], f"{API}/{path}?limit=1000"
    while url:
        body, headers = _get(url)
        for it in json.loads(body.decode("utf-8")):
            if it.get("type") == "file":
                out.append((it["path"], int(it.get("size") or 0)))
        link = headers.get("Link") or headers.get("link") or ""
        url = None
        if 'rel="next"' in link:
            seg = [s for s in link.split(",") if 'rel="next"' in s][0]
            url = seg.split(";")[0].strip().strip("<>")
    return out


def existing_stems():
    stems = set()
    for sub in ("train", "val"):
        for p in (HAVE / "images" / sub).glob("*.jpg"):
            stems.add(p.stem)
        for p in (HAVE / "images" / sub).glob("*.png"):
            stems.add(p.stem)
    return stems


def fetch(path: str, size: int, retries=3):
    dst = DEST / path.replace("data/", "", 1)
    if dst.exists() and dst.stat().st_size == size:
        return dst, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            body, _ = _get(f"{RAW}/{path}", timeout=180)
            if size and len(body) != size:
                raise IOError("size mismatch")
            dst.write_bytes(body)
            return dst, len(body)
        except Exception:  # noqa: BLE001
            time.sleep(1.2 * (i + 1))
    return dst, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    have = existing_stems()
    tasks = []
    for split in a.splits:
        imgs = list_dir(f"data/images/{split}")
        labs = {Path(p).stem: (p, s) for p, s in list_dir(f"data/labels/{split}")}
        new = [(p, s) for p, s in imgs if Path(p).stem not in have and Path(p).stem in labs]
        print(f"[{split}] 远端 {len(imgs)} 张，本地已有 {len(imgs) - len(new)}，待下 {len(new)}")
        for p, s in new:
            tasks.append((p, s))
            tasks.append(labs[Path(p).stem])
    mb = sum(s for _, s in tasks) / 1024**2
    print(f"合计 {len(tasks)} 个文件 / {mb:.1f} MB")
    if a.list:
        return

    t0 = time.time()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, p, s): p for p, s in tasks}
        for f in as_completed(futs):
            dst, n = f.result()
            done += 1
            if n < 0:
                failed.append(str(dst))
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6) / 60
                print(f"\r  {done}/{len(tasks)}  {el/60:.1f} min  {rate:.1f} 文件/s  ETA {eta:.0f} min", end="", flush=True)
    print()
    man = {
        "files_ok": done - len(failed),
        "failed": failed[:30],
        "mb": round(mb, 1),
        "minutes": round((time.time() - t0) / 60, 1),
        "note": "下完必须重跑 scripts/build_gwhd_clean.py 筛查 test 同源图",
    }
    (DEST / "manifest_more.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
