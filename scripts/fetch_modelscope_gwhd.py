# -*- coding: utf-8 -*-
"""从 ModelScope（魔搭）下载 GWHD2021 全量包（中国境内可访问，HF 目前被墙/限流）。

包信息：swordoffer/global-wheat-challenge-2021 → gwhd_2021.zip ≈ 10.2 GB
内容：6512 张 1024×1024 田间图 + 标注（约 30 万个麦穗头），比我们现有的 GWHD2020 子集更多样。

实现要点：
  - Range 分段 + 多线程并行（默认 8 段），比单连接快数倍；
  - 每段独立落盘 `*.part<N>`，中断重跑自动跳过已完成段（断点续传）；
  - 全部完成后按顺序拼接成 zip。

用法:
  python scripts/fetch_modelscope_gwhd.py --list        # 只看大小
  python scripts/fetch_modelscope_gwhd.py --segments 8  # 下载
"""
import argparse
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data_external" / "gwhd2021"
REPO = "swordoffer/global-wheat-challenge-2021"
FILE = "gwhd_2021.zip"
URL = f"https://www.modelscope.cn/api/v1/datasets/{REPO}/repo?Revision=master&FilePath={FILE}"
UA = {"User-Agent": "Mozilla/5.0"}
CTX = ssl.create_default_context(cafile=str(ROOT / "pkgs" / "certifi" / "cacert.pem"))


def total_size() -> int:
    req = urllib.request.Request(URL, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        cr = r.headers.get("Content-Range") or ""
    return int(cr.split("/")[-1]) if "/" in cr else 0


def worker(url: str, start: int, end: int, part: Path, idx: int, retries=4):
    if part.exists() and part.stat().st_size == (end - start + 1):
        print(f"  [seg{idx}] 已完成，跳过")
        return True
    for attempt in range(retries):
        try:
            done = part.stat().st_size if part.exists() else 0
            s = start + done
            if s > end:
                return True
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{end}"})
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r, open(part, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if part.stat().st_size == (end - start + 1):
                print(f"  [seg{idx}] 完成 {part.stat().st_size/1024**2:.0f} MB")
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  [seg{idx}] 第 {attempt+1} 次失败: {type(e).__name__} {str(e)[:60]}")
            time.sleep(2 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    size = total_size()
    if not size:
        raise SystemExit("拿不到文件大小，检查网络/URL")
    print(f"[info] {FILE}: {size/1024**3:.2f} GB  ({size} bytes)")
    if a.list:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [OUT_DIR / f"{FILE}.part{i}" for i in range(a.segments)]
    seg = size // a.segments
    ranges = []
    for i in range(a.segments):
        s = i * seg
        e = size - 1 if i == a.segments - 1 else (i + 1) * seg - 1
        ranges.append((s, e))

    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.segments) as ex:
        futs = [ex.submit(worker, URL, s, e, parts[i], i) for i, (s, e) in enumerate(ranges)]
        ok = all(f.result() for f in futs)
    if not ok:
        raise SystemExit("有分段失败，重跑本脚本会续传")

    dst = OUT_DIR / FILE
    with open(dst, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    got = dst.stat().st_size
    print(f"[done] {dst}  {got/1024**3:.2f} GB  用时 {(time.time()-t0)/60:.1f} min")
    if got != size:
        raise SystemExit(f"大小不符：{got} != {size}")
    for p in parts:
        p.unlink(missing_ok=True)
    (OUT_DIR / "download_manifest.json").write_text(
        json.dumps({"file": str(dst), "bytes": got, "minutes": round((time.time() - t0) / 60, 1)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
