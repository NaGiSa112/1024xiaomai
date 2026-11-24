# -*- coding: utf-8 -*-
"""单连接流式下载 ModelScope 上的 GWHD2021 全量包（8 线程分段会被限流，实测单流更快）。

- 单连接实测 3.79 MB/s（9.51 GB ≈ 42 min）
- 断点续传：中断后重跑，从已下载字节数继续（Range 请求）
- 每 512 MB 打印一次进度与 ETA

用法:
  python scripts/fetch_ms_single.py
  python scripts/fetch_ms_single.py --url <自定义> --out <路径>
"""
import argparse
import ssl
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://www.modelscope.cn/api/v1/datasets/swordoffer/global-wheat-challenge-2021/repo"
    "?Revision=master&FilePath=gwhd_2021.zip"
)
CTX = ssl.create_default_context(cafile=str(ROOT / "pkgs" / "certifi" / "cacert.pem"))
UA = {"User-Agent": "Mozilla/5.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(ROOT / "data_external" / "gwhd2021" / "gwhd_2021.zip"))
    a = ap.parse_args()

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 先取总大小
    req = urllib.request.Request(a.url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        cr = r.headers.get("Content-Range") or ""
    total = int(cr.split("/")[-1]) if "/" in cr else 0
    have = dst.stat().st_size if dst.exists() else 0
    print(f"[info] 总大小 {total/1024**3:.2f} GB，已有 {have/1024**3:.2f} GB", flush=True)
    if total and have >= total:
        print("[done] 已下载完整，无需继续", flush=True)
        return

    t0 = time.time()
    mark = have
    while total == 0 or have < total:
        headers = {**UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(a.url, headers=headers)
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r, open(dst, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have - mark >= 512 * 1024**2:
                        el = time.time() - t0
                        rate = (have - (dst.stat().st_size - (have - mark))) / max(el, 1e-6)
                        speed = (have - mark) / el / 1024**2
                        eta = (total - have) / max(have - mark, 1) * el / 60
                        print(
                            f"  {have/1024**3:.2f}/{total/1024**3:.2f} GB  "
                            f"{speed:.2f} MB/s  ETA {eta:.0f} min",
                            flush=True,
                        )
                        mark = have
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 中断: {type(e).__name__} {str(e)[:80]}，3 秒后续传", flush=True)
            time.sleep(3)
            have = dst.stat().st_size if dst.exists() else have
            continue
        if total == 0:
            break

    got = dst.stat().st_size
    print(f"[done] {dst}  {got/1024**3:.2f} GB  用时 {(time.time()-t0)/60:.1f} min", flush=True)
    if total and got != total:
        print(f"[warn] 大小不符 {got} != {total}", flush=True)


if __name__ == "__main__":
    main()
