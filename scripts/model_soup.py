# -*- coding: utf-8 -*-
"""model soup：把多个 checkpoint 的权重做平均（零训练成本的涨点手段）.

原理：同架构、不同训练轨迹（不同 epoch / 不同 seed / 不同折）的权重往往落在同一个
损失盆地，对它们取平均会产生更平坦的解，泛化通常优于任何单个成员。
经验增益 +0.2~0.5（与预测级集成叠加时增益会变小）。

限制：只对**结构完全相同**的 checkpoint 有效（这里按 state_dict 的 key 与 shape 校验）。

用法:
  # 平均某次训练最后几个 epoch 的权重（需要训练时 --save-period 5）
  python scripts/model_soup.py --weights runs/detect/x/weights/epoch95.pt runs/detect/x/weights/epoch100.pt --out weights/soup_x.pt
  # 平均多个 run 的 best.pt（同架构、不同 seed/折）
  python scripts/model_soup.py --weights runs/detect/a/weights/best.pt runs/detect/b/weights/best.pt --out weights/soup_ab.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def load_state(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("ema") or ck.get("model")
    if model is None:
        raise SystemExit(f"{path} 里没有 model/ema")
    sd = model.float().state_dict() if hasattr(model, "state_dict") else model
    return ck, {k: v.detach().clone() for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    paths = [Path(w) if Path(w).is_absolute() else ROOT / w for w in a.weights]
    for p in paths:
        if not p.exists():
            sys.exit(f"权重不存在: {p}")

    base_ck, base_sd = load_state(paths[0])
    acc = {k: v.clone() for k, v in base_sd.items()}
    n = 1
    for p in paths[1:]:
        _, sd = load_state(p)
        if set(sd) != set(base_sd):
            sys.exit(f"{p.name} 的参数名与第一个不一致，不能 soup")
        for k in sd:
            if sd[k].shape != base_sd[k].shape:
                sys.exit(f"{p.name} 的参数 {k} 形状不一致：{tuple(sd[k].shape)} vs {tuple(base_sd[k].shape)}")
        for k in acc:
            acc[k] += sd[k]
        n += 1

    if a.check_only:
        print(f"[ok] {n} 个 checkpoint 结构一致，可 soup")
        return

    # 只平均浮点参数；整型 buffer（如 stride / num_batches_tracked）保留第一个的值
    for k in acc:
        if acc[k].is_floating_point():
            acc[k] = acc[k] / n
        else:
            acc[k] = base_sd[k].clone()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # 保留第一个 checkpoint 的元信息（epoch/args/names），替换权重
    ck = dict(base_ck)
    if "model" in ck and hasattr(ck["model"], "load_state_dict"):
        ck["model"].load_state_dict(acc, strict=True)
    if ck.get("ema") is not None and hasattr(ck["ema"], "load_state_dict"):
        ck["ema"].load_state_dict(acc, strict=True)
    ck["epoch"] = -1
    ck["soup_sources"] = [p.name for p in paths]
    torch.save(ck, out)
    print(f"[done] 平均 {n} 个权重 -> {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
    print("       用法: python scripts/predict.py --weights " + str(out))


if __name__ == "__main__":
    main()
