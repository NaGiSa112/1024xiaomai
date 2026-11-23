# -*- coding: utf-8 -*-
"""5 折 OOF 评测：把各折 val 的预测拼成"每张图都有且仅有一个折外预测"，算 mAP@0.5。

相比 100 张固定 val，OOF 用的是全部 900 张（每张都被 1 个模型预测过），
指标噪声从 σ≈0.005 降到 σ≈0.002，选型更可靠。

用法:
  python scripts/oof_eval.py --pred-template "preds/oof_fold{fold}" --folds 0 1 2 3 4
  # 每个 pred 目录里是 <stem>.txt（提交格式），图片清单来自 data/folds/fold{fold}_val.txt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from eval_submission import average_precision, iou_matrix, load_gt, load_pred  # noqa: E402


def gt_for(img: Path) -> Path:
    """.../images/<split>/xxx.png -> .../labels/<split>/xxx.txt"""
    parts = list(img.parts)
    assert "images" in parts, f"图片路径里没有 images: {img}"
    parts[len(parts) - 3] = "labels"   # images -> labels，保留 split 层级
    return Path(*parts).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-template", default="preds/oof_fold{fold}")
    ap.add_argument("--folds-dir", default="data/folds")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    per_fold = []
    all_tp, all_conf, n_gt_total = [], [], 0

    for k in a.folds:
        lp = ROOT / a.folds_dir / f"fold{k}_val.txt"
        imgs = [Path(l.strip()) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        pred_dir = ROOT / a.pred_template.format(fold=k)

        tp_flags, confs, n_gt = [], [], 0
        for img in imgs:
            g = load_gt(gt_for(img))
            p, c = load_pred(pred_dir / f"{img.stem}.txt")
            n_gt += len(g)
            if len(p) == 0:
                continue
            ious = iou_matrix(p, g)
            order = np.argsort(-c)
            matched = np.zeros(len(g), dtype=bool)
            for idx in order:
                j = int(np.argmax(ious[idx])) if len(g) else 0
                if len(g) and ious[idx, j] >= a.iou and not matched[j]:
                    matched[j] = True
                    tp_flags.append(1)
                else:
                    tp_flags.append(0)
                confs.append(c[idx])

        if not tp_flags:
            print(f"fold{k}: 没有预测框，跳过")
            continue
        tp_flags = np.array(tp_flags)
        confs = np.array(confs)
        order = np.argsort(-confs)
        tp = np.cumsum(tp_flags[order])
        fp = np.cumsum(1 - tp_flags[order])
        rec = tp / max(n_gt, 1)
        prec = tp / np.maximum(tp + fp, 1e-12)
        mAP = average_precision(rec, prec)
        per_fold.append({"fold": k, "images": len(imgs), "gt_boxes": int(n_gt),
                         "pred_boxes": int(len(confs)), "mAP@0.5": round(float(mAP), 5)})
        print(f"fold{k}: imgs={len(imgs)} gt={n_gt} pred={len(confs)} mAP@0.5={mAP:.5f}")

        all_tp.append(tp_flags)
        all_conf.append(confs)
        n_gt_total += n_gt

    if all_tp:
        tp_flags = np.concatenate(all_tp)
        confs = np.concatenate(all_conf)
        order = np.argsort(-confs)
        tp = np.cumsum(tp_flags[order])
        fp = np.cumsum(1 - tp_flags[order])
        rec = tp / max(n_gt_total, 1)
        prec = tp / np.maximum(tp + fp, 1e-12)
        oof = float(average_precision(rec, prec))
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        b = int(np.argmax(f1))
        summary = {
            "folds": per_fold,
            "oof_images": int(sum(f["images"] for f in per_fold)),
            "oof_gt_boxes": int(n_gt_total),
            "OOF_mAP@0.5": round(oof, 5),
            "oof_mean_of_folds": round(float(np.mean([f["mAP@0.5"] for f in per_fold])), 5),
            "oof_std_of_folds": round(float(np.std([f["mAP@0.5"] for f in per_fold])), 5),
            "best_f1": round(float(f1[b]), 4),
            "best_f1_conf": round(float(confs[order][b]), 4),
        }
        out = ROOT / "preds" / "oof_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
