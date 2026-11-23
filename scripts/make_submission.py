# -*- coding: utf-8 -*-
"""把 predict.py 产出的 txt 目录打包成提交用的 submit.zip.

赛题要求(2026 小麦麦穗目标检测挑战赛):
  zip 内含 submit 文件夹, 文件夹内是与测试集图片同名的 txt (UTF-8),
  每行: <class_id> <x_center> <y_center> <width> <height> <confidence>

用法:
  python scripts/make_submission.py --pred preds/test_y11s_1024 --tag y11s_1024
  # -> submissions/submit_y11s_1024.zip   (内含 submit/0001.txt ... submit/0100.txt)
  python scripts/make_submission.py --pred preds/test_x --tag x --check data/dataset/images/test
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="predict.py 输出目录(含 *.txt)")
    p.add_argument("--tag", required=True, help="提交文件名后缀")
    p.add_argument("--check", default="data/dataset/images/test", help="测试图目录, 用来核对文件名齐全")
    p.add_argument("--out", default="submissions")
    return p.parse_args()


def validate(txt: Path, problems: list):
    raw = txt.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        problems.append(f"{txt.name}: 带 BOM, 要求 UTF-8 无 BOM")
    if b"\r" in raw:
        # CRLF 在部分平台的解析器里会把 '\r' 粘到 confidence 字段上, 统一转成 LF
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        txt.write_bytes(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        problems.append(f"{txt.name}: 非 UTF-8 ({e})")
        return 0
    n = 0
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            problems.append(f"{txt.name}:{i} 字段数={len(parts)} (应为 6)")
            continue
        try:
            k = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            problems.append(f"{txt.name}:{i} 无法解析: {line!r}")
            continue
        if k != 0:
            problems.append(f"{txt.name}:{i} class_id={k} (本赛题只有 0)")
        xc, yc, w, h, c = vals
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            problems.append(f"{txt.name}:{i} 坐标越界: {line!r}")
        if not (0 <= c <= 1):
            problems.append(f"{txt.name}:{i} confidence 越界: {line!r}")
        n += 1
    return n


def main():
    a = parse_args()
    pred = Path(a.pred)
    if not pred.is_absolute():
        pred = ROOT / pred
    if not pred.exists():
        sys.exit(f"预测目录不存在: {pred}")

    txts = sorted(pred.glob("*.txt"))
    if not txts:
        sys.exit(f"没有 txt: {pred}")

    problems: list[str] = []
    total_boxes = 0
    for t in txts:
        total_boxes += validate(t, problems)

    # 文件名必须与测试图一一对应
    if a.check:
        img_dir = Path(a.check)
        if not img_dir.is_absolute():
            img_dir = ROOT / img_dir
        stems_img = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        stems_txt = {t.stem for t in txts}
        missing = sorted(stems_img - stems_txt)
        extra = sorted(stems_txt - stems_img)
        if missing:
            problems.append(f"缺 {len(missing)} 个 txt: {missing[:5]}")
        if extra:
            problems.append(f"多出 {len(extra)} 个 txt: {extra[:5]}")
        print(f"[check] 图片 {len(stems_img)} / txt {len(stems_txt)}")

    print(f"[check] 目标框总数 {total_boxes}")
    if problems:
        print(f"[!!] 发现 {len(problems)} 个格式问题(前 20 条):")
        for p in problems[:20]:
            print("   -", p)
        sys.exit(1)
    print("[check] 格式校验通过")

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"submit_{a.tag}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for t in txts:
            z.write(t, arcname=f"submit/{t.name}")

    info = {
        "zip": str(zip_path),
        "files": len(txts),
        "boxes": total_boxes,
        "size_mb": round(zip_path.stat().st_size / 1024**2, 2),
    }
    (out_dir / f"submit_{a.tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n提交文件: {zip_path}")


if __name__ == "__main__":
    main()
