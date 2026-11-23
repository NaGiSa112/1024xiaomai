# -*- coding: utf-8 -*-
"""赛题数据 vs GWHD 同源核查（dHash 召回 + 像素级确认）.

只看 dHash 会误判：麦田纹理高度相似，64 bit 感知哈希在"距离<=6"这一档假阳性很多。
所以这里两段式：dHash 取 top-k 候选 -> 缩图算像素 MAD -> MAD<6/255 才认定为同源重编码。

用法: python scripts/check_overlap.py
输出: docs/重叠检查.md
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
GWH = ROOT / "data_external" / "gwhd"
OUT_MD = ROOT / "docs" / "重叠检查.md"


def dhash(p: Path, size=16) -> np.uint64:
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)[:64]
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def thumb(p: Path, n=128) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize((n, n), Image.BILINEAR), dtype=np.int16)


def popcount_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """汉明距离矩阵。用移位+查表逐字节 popcount，避免 ndarray.view 的坑。"""
    a = np.ascontiguousarray(np.atleast_1d(a).astype(np.uint64))
    b = np.ascontiguousarray(np.atleast_1d(b).astype(np.uint64))
    x = np.bitwise_xor(a[:, None], b[None, :])
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    out = np.zeros(x.shape, dtype=np.uint16)
    for shift in range(0, 64, 8):
        out += lut[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return out


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    splits = {s: sorted((DATA / "images" / s).glob("*.png")) for s in ("train", "val", "test")}
    g_imgs = sorted((GWH / "images" / "train").glob("*.jpg")) + sorted((GWH / "images" / "val").glob("*.jpg"))

    print(f"GWHD 参照库: {len(g_imgs)} 张")
    g_hash = np.array([dhash(p) for p in g_imgs], dtype=np.uint64)
    g_md5 = {md5(p): p for p in g_imgs}

    report = {"gwhd_library": len(g_imgs), "splits": {}}
    for split, imgs in splits.items():
        exact = 0
        dups = []
        for p in imgs:
            if md5(p) in g_md5:
                exact += 1
            h = dhash(p)
            d = popcount_matrix(np.array([h], dtype=np.uint64), g_hash).ravel()
            cand = np.argsort(d)[:3]          # 取 dHash 最近的 3 个再验像素
            t = thumb(p)
            best = (999.0, None, 999)
            for j in cand:
                m = float(np.abs(t - thumb(g_imgs[j])).mean())
                if m < best[0]:
                    best = (m, g_imgs[j], int(d[j]))
            if best[0] < 6.0:                 # 像素 MAD < 6/255 视为同一张图的重编码
                dups.append({"img": p.name, "gwhd": best[1].name, "mad": round(best[0], 2), "dhash": best[2]})
        report["splits"][split] = {
            "total": len(imgs),
            "exact_md5": exact,
            "same_image_mad<6": len(dups),
            "ratio": round(len(dups) / max(len(imgs), 1), 3),
            "examples": dups[:10],
        }
        print(f"{split}: total={len(imgs)} md5同文件={exact} 同图重编码={len(dups)} ({len(dups)/max(len(imgs),1):.1%})")

    lines = [
        "# 赛题数据 vs GWHD 同源核查",
        "",
        f"GWHD 参照库（本机）：**{len(g_imgs)}** 张 1024×1024",
        "",
        "判定口径：dHash 取最近 3 个候选 → 缩图 128×128 算像素 MAD → **MAD < 6/255 认定为同一张图**"
        "（PNG/JPEG 重编码级别的差异；不同图 MAD 通常 > 30）。",
        "",
        "| 划分 | 总数 | 字节完全相同 | 同图（重编码） | 占比 |",
        "|---|---|---|---|---|",
    ]
    for s, v in report["splits"].items():
        lines.append(f"| {s} | {v['total']} | {v['exact_md5']} | {v['same_image_mad<6']} | {v['ratio']:.1%} |")
    lines += ["", "## 结论", ""]
    tr = report["splits"]["train"]["ratio"]
    va = report["splits"]["val"]["ratio"]
    te = report["splits"]["test"]["ratio"]
    lines += [
        f"- 训练集有 **{tr:.0%}** 的图能在 GWHD 里找到同一张图的重编码版本，验证集 **{va:.0%}**，测试集 **{te:.0%}**。",
        "- 说明赛题数据与 GWHD **同源同分布**（同一批 1024×1024 田间图），不是「相似领域」。",
        "- 但**标注不同**：同一张图，赛题标注框数与本机 GWHD 标注框数不一致"
        "（例如 train/0003 赛题 64 框 vs GWHD 53 框）。⇒ 外部数据的**标签口径有差异**，"
        "适合用来预训练学特征，最后一律在赛题 900 张上微调对齐口径。",
        "- **合规红线**：与 test 同源的那些 GWHD 图（及其标签）**一律排除出训练集**，"
        "不使用任何与测试图对应的标注，避免「测试集参与训练」。详见 `scripts/build_gwhd_clean.py`。",
        "",
    ]
    if report["splits"]["test"]["examples"]:
        lines.append("## 测试集同源样例（仅用于合规排除，不用于训练/评测）")
        lines += [f"- `test/{e['img']}` ↔ GWHD `{e['gwhd']}`（MAD {e['mad']}）" for e in report["splits"]["test"]["examples"]]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "overlap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出 docs/重叠检查.md, docs/overlap.json")


if __name__ == "__main__":
    main()
