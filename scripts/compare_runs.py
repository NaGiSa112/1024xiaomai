# -*- coding: utf-8 -*-
"""比较两个（或多个）训练 run 的 best mAP50 / mAP50-95，用来做门禁判断。

用法:
  python scripts/compare_runs.py --runs gate_A_n900 gate_B_n900
  python scripts/compare_runs.py --runs long_yolo11l_1024 long_yolo11m_1024 fold0_y11s_1024
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_best(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = [r for r in rows if r.get("metrics/mAP50(B)") not in (None, "", "nan")]
    if not rows:
        return None

    def f(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    best = max(rows, key=lambda r: f(r, "metrics/mAP50-95(B)"))
    last = rows[-1]
    return {
        "run": run_dir.name,
        "epochs_done": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_mAP50": round(f(best, "metrics/mAP50(B)"), 5),
        "best_mAP50_95": round(f(best, "metrics/mAP50-95(B)"), 5),
        "last_mAP50": round(f(last, "metrics/mAP50(B)"), 5),
        "last_mAP50_95": round(f(last, "metrics/mAP50-95(B)"), 5),
        "hours": round(f(last, "time") / 3600, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    out = []
    for name in a.runs:
        p = ROOT / "runs" / "detect" / name
        r = read_best(p)
        if r:
            out.append(r)
        else:
            print(f"[warn] {name}: 没有 results.csv 或还没有有效指标")
    if not out:
        return
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if len(out) >= 2:
        base = out[0]
        for r in out[1:]:
            print(f"{r['run']} vs {base['run']}:  mAP50 {r['best_mAP50'] - base['best_mAP50']:+.4f}"
                  f"   mAP50-95 {r['best_mAP50_95'] - base['best_mAP50_95']:+.4f}")


if __name__ == "__main__":
    main()
