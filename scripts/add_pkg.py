# -*- coding: utf-8 -*-
"""add_pkg.py -- 本机 pip 不可用时的替代安装器.

背景: DSH 沙箱禁止在临时目录里删除/重命名已解包的 wheel, pip 因此必然失败:
  ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied:
         ...\\Temp\\dsh-xxxx\\pip-unpack-xxxx\\xxx.whl
所以这里绕开 pip: 直接查 PyPI 索引 -> 下载 wheel -> 解包进 pkgs\\.

用法:
  python scripts/add_pkg.py onnx onnxruntime
  python scripts/add_pkg.py "timm>=1.0"
  python scripts/add_pkg.py --index https://pypi.tuna.tsinghua.edu.cn pep8-naming

解出来的包装到 pkgs\\(与 env.ps1 里的 PYTHONPATH 一致), 依赖会递归解析.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGS = ROOT / "pkgs"
WHEELS = ROOT / "_wheels"
MANIFEST = ROOT / "pkgs_manifest.json"

PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
sys.path.insert(0, str(PKGS))

DEFAULT_INDEXES = [
    "https://pypi.tuna.tsinghua.edu.cn",
    "https://pypi.org",
]

# 平台化 wheel 标签优先级 (win_amd64), 元组 = (python tag, abi tag, platform tag)
TAG_PRIORITY = [
    (PY_TAG, PY_TAG, "win_amd64"),
    (PY_TAG, "abi3", "win_amd64"),
    (PY_TAG, "none", "win_amd64"),
    (PY_TAG, "none", "any"),
    ("py3", "none", "any"),
    ("py2.py3", "none", "any"),
]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def vkey(ver: str):
    parts = []
    for p in re.split(r"[.\-+]", ver):
        m = re.match(r"^(\d+)(.*)$", p)
        parts.append((int(m.group(1)) if m else -1, m.group(2) if m else p))
    return tuple(parts)


def is_prerelease(ver: str) -> bool:
    return bool(re.search(r"(a|b|rc|dev|alpha|beta|pre)\d*$", ver, re.I))


def _retry(fn, *args, tries=3, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  网络偶发 SSL EOF
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "wheat-add-pkg/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "wheat-add-pkg/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def choose_wheel(files):
    """在 release 的 files 里挑最合适的 win/CPython 3.12 wheel."""
    best = None
    for f in files:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        stem = fn[: -len(".whl")]
        try:
            _, py, abi, plat = stem.rsplit("-", 3)
        except ValueError:
            continue
        score = None
        for i, (pytag, abi_tag, plat_tag) in enumerate(TAG_PRIORITY):
            if plat != plat_tag or abi != abi_tag:
                continue
            if pytag not in py.split("."):
                continue
            score = i
            break
        if score is None:
            continue
        cand = (score, f["url"], fn)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


def release_of(name: str, spec: str | None, indexes):
    last = None
    for base in indexes:
        try:
            data = fetch_json(f"{base}/pypi/{name}/json")
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        releases = data.get("releases", {})
        cands = []
        for ver, files in releases.items():
            if not files:
                continue
            if is_prerelease(ver):
                continue
            if spec and not satisfies(ver, spec):
                continue
            pick = choose_wheel(files)
            if pick:
                cands.append((vkey(ver), ver, pick))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        _, ver, pick = cands[-1]
        return ver, pick[1], pick[2]
    if last:
        print(f"  [warn] {name}: {last}")
    return None, None, None


def satisfies(ver: str, spec: str) -> bool:
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = re.match(r"^(>=|<=|==|!=|~=|>|<)?\s*([0-9][\w.\-]*)$", clause)
        if not m:
            continue
        op, target = m.group(1) or "==", m.group(2)
        a, b = vkey(ver), vkey(target)
        ok = {
            ">=": a >= b,
            "<=": a <= b,
            "==": a == b,
            "!=": a != b,
            ">": a > b,
            "<": a < b,
            "~=": a >= b and a[: len(b) - 1] == b[: len(b) - 1],
        }[op]
        if not ok:
            return False
    return True


def parse_spec(word: str):
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", word.strip())
    if not m:
        return None, None
    name, rest = m.group(1), m.group(2).strip()
    if rest.startswith("["):  # extras，忽略
        rest = rest.split("]", 1)[1].strip()
    spec = rest.lstrip("=<>!~") or None
    if rest and rest[0] in "=<>!~":
        spec = rest
    return norm(name), spec


def deps_from_metadata(whl_path: Path):
    out = []
    with zipfile.ZipFile(whl_path) as z:
        meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if not meta:
            return out
        text = z.read(meta[0]).decode("utf-8", "replace")
    for line in text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        val = line.split(":", 1)[1].strip()
        if "extra ==" in val:  # 只装主依赖
            continue
        req = val.split(";")[0].strip()
        out.append(req)
    return out


def installed(name: str):
    """返回 pkgs 里已安装的 dist-info 目录名.

    注意: dist-info 名形如 `numpy-2.5.4.dist-info`, 必须先去掉 `.dist-info`
    再 rsplit('-', 1) 才是包名; 否则 `norm('numpy-2.5.4') != 'numpy'`,
    已装包会被误判为未安装并**降级覆盖**(踩过: numpy 2.5.4 被 pyarrow 的
    依赖解析降成 2.0.1)。
    """
    for d in PKGS.glob("*.dist-info"):
        base = d.name[: -len(".dist-info")] if d.name.endswith(".dist-info") else d.name
        parts = base.rsplit("-", 1)
        pkg = parts[0] if len(parts) == 2 else base
        if norm(pkg) == norm(name):
            yield d.name


def install_one(name: str, spec: str | None, indexes, done: dict):
    key = norm(name)
    if key in done:
        return
    for d in installed(key):
        ver = d.split("-")[-1]
        if spec is None or satisfies(ver, spec):
            print(f"  [skip] {key} {ver} 已存在")
            done[key] = ver
            return

    ver, url, fn = release_of(key, spec, indexes)
    if not url:
        print(f"  [fail] {key}: 没有可用的 wheel ({PY_TAG}/win_amd64/any)")
        done[key] = None
        return

    WHEELS.mkdir(parents=True, exist_ok=True)
    dest = WHEELS / fn
    if not dest.exists():
        print(f"  [down] {fn}")
        try:
            dest.write_bytes(_retry(fetch_bytes, url))
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {key}: 下载失败 {e}")
            done[key] = None
            return
    with zipfile.ZipFile(dest) as z:
        z.extractall(PKGS)
    print(f"  [ok]   {key} {ver} -> pkgs")
    done[key] = ver

    for req in deps_from_metadata(dest):
        dn, ds = parse_spec(req)
        if dn:
            try:
                install_one(dn, ds, indexes, done)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] 依赖 {dn} 安装失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+")
    ap.add_argument("--index", action="append", default=None)
    a = ap.parse_args()
    indexes = a.index or DEFAULT_INDEXES

    PKGS.mkdir(exist_ok=True)
    done: dict[str, str | None] = {}
    for word in a.packages:
        name, spec = parse_spec(word)
        if not name:
            print(f"[warn] 无法解析: {word}")
            continue
        print(f"[pkg] {name}{spec or ''}")
        install_one(name, spec, indexes, done)

    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        man = {}
    man.update({k: v for k, v in done.items() if v})
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n写入清单: {MANIFEST} ({len(man)} 项)")


if __name__ == "__main__":
    main()
