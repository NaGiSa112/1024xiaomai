# -*- coding: utf-8 -*-
"""从 runs/detect/*/results.csv 生成实验记录表（docs/实验记录.md）。

每次跑完训练后重跑一次，直接得到可以贴进报告/README 的表格：
  python scripts/make_experiment_log.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "detect"
OUT = ROOT / "docs" / "实验记录.md"

# 说明文字（按 run 名匹配，人工维护）——不影响表格数字
NOTES = {
    "smoke_n_640": "链路验证，2 epoch",
    "probe_s_1024_e3": "3 epoch 快速探针，确认 1024 分辨率收益",
    "fold0_y11s_1024": "5 折第 1 折，**干净折外**（720 训练 / 182 折外）",
    "fold1_y11s_1024_partial_ep20": "5 折第 2 折，中途让出 GPU",
    "ref800_y11s_1024_partial_ep8": "官方 800/100 同口径对照，中途让出 GPU",
    "long_pre_y11l_768": "GWHD-clean(2834) 同源预训练，**未接触赛题数据**",
    "long_y11l_1024": "预训练->900 全量微调（val 含污染，仅看收敛）",
    "x_pre_gwhd_768": "yolo11x 同源预训练",
    "x900_1024": "yolo11x 900 全量微调（val 含污染）",
    "l1280_900": "yolo11l @1280，分辨率维度成员",
    "m900_1024": "yolo11m，容量维度成员",
}


def fnum(r, key, default=0.0):
    try:
        v = r.get(key)
        return float(v) if v not in (None, "", "nan") else default
    except (TypeError, ValueError):
        return default


def read_run(d: Path):
    cs = d / "results.csv"
    if not cs.exists():
        return None
    rows = [r for r in csv.DictReader(open(cs, encoding="utf-8")) if r.get("epoch")]
    if not rows:
        return None
    best = max(rows, key=lambda r: fnum(r, "metrics/mAP50-95(B)"))
    model = data = ""
    args = d / "args.yaml"
    if args.exists():
        t = args.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^model:\s*(\S+)", t, re.M)
        md = re.search(r"^data:\s*(\S+)", t, re.M)
        if m:
            model = Path(m.group(1).replace("\\", "/")).name
        if md:
            data = Path(md.group(1).replace("\\", "/")).name
    return {
        "run": d.name,
        "model": model,
        "data": data,
        "epochs_done": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "mAP50": fnum(best, "metrics/mAP50(B)"),
        "mAP50_95": fnum(best, "metrics/mAP50-95(B)"),
        "P": fnum(best, "metrics/precision(B)"),
        "R": fnum(best, "metrics/recall(B)"),
        "hours": fnum(rows[-1], "time") / 3600,
    }


def main():
    runs = [r for r in (read_run(d) for d in sorted(RUNS.iterdir()) if d.is_dir()) if r]
    runs.sort(key=lambda r: r["mAP50"], reverse=True)

    lines = [
        "# 实验记录（自动生成）",
        "",
        "> 由 `python scripts/make_experiment_log.py` 从 `runs/detect/*/results.csv` 生成。",
        "> 指标取每个 run 的 **best epoch**（按 mAP50-95 选）。",
        "> ⚠️ 凡是用 `wheat_full900.yaml`（900 张含官方 val）训练的 run，其 val 指标**含污染**，"
        "只能看收敛趋势，不能当泛化性能；干净口径看 `fold0_y11s_1024` 与 `long_pre_y11l_768`。",
        "",
        "| run | 初始权重 | 数据 | epoch | best ep | mAP50 | mAP50-95 | P | R | 耗时(h) | 说明 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        note = NOTES.get(r["run"], "")
        lines.append(
            f"| `{r['run']}` | {r['model']} | {r['data']} | {r['epochs_done']} | {r['best_epoch']} | "
            f"**{r['mAP50']:.4f}** | {r['mAP50_95']:.4f} | {r['P']:.4f} | {r['R']:.4f} | {r['hours']:.2f} | {note} |"
        )

    lines += [
        "",
        "## 关键结论速览",
        "",
        "| 证据 | 数值 | 口径 |",
        "|---|---|---|",
    ]
    clean = [r for r in runs if r["run"] in ("fold0_y11s_1024", "long_pre_y11l_768")]
    for r in clean:
        lines.append(f"| `{r['run']}` | mAP50 {r['mAP50']:.4f} | **干净** |")
    lines += [
        "",
        "- 赛题数据与 GWHD **同源**（同图重编码占比 train 54.2% / val 58.0% / test 47.0%），"
        "见 `docs/重叠检查.md`；训练集已剔除 539 张同源图（含 47 张 test 同源），见 `scripts/build_gwhd_clean.py`。",
        "- 纯 GWHD 预训练（未接触赛题数据）就能在我们的 val 上到 mAP50 ≈ 0.88，"
        "这是「同源迁移有效」的最直接证据。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\n写出 {OUT}（{len(runs)} 个 run）")


if __name__ == "__main__":
    main()
