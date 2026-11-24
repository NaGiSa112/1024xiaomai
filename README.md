# 小麦麦穗目标检测（Wheat-SOD）· 讯飞算法赛

单类别目标检测：田间实拍 **1024×1024** 小麦图像，识别并定位麦穗（`wheat_ear`）。
官方指标 **mAP@0.5**（每天最多 3 次提交，取历史最优）。

- 赛题：<https://challenge.xfyun.cn/topic/info?type=Wheat-SOD-train>
- 数据：train 800 / val 100 / test 100（test 无标注，仅推理）
- 任务难点：密集**中小目标** —— 平均 42.7 框/图（最多 190），框宽中位数 0.068（1024 图上约 **68 px**）

---

## 1. 结果

### 1.1 干净口径（可信）

| 实验 | 模型 | 数据 | mAP@0.5 | mAP@0.5:0.95 | 说明 |
|---|---|---|---|---|---|
| 5 折第 1 折 | yolo11s @1024 | 720 训练 / 182 折外 | **0.9108** | 0.5135 | 无外部数据、无预训练的诚实基线 |
| GWHD 同源预训练 | yolo11l @768 | GWHD-clean 2834 张 | **0.8812** | 0.4596 | **未接触任何赛题数据**，仅靠外部数据 |

### 1.2 含污染（900 全量训练，val 被训练过，只看收敛）

| 实验 | 配置 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 主力模型 | yolo11l @1024，预训练→900 微调 140 ep | 0.9880 | 0.7011 |

完整实验表见 [`docs/实验记录.md`](docs/实验记录.md)（由 `scripts/make_experiment_log.py` 自动生成）。

### 1.3 线上提交

见 [`docs/提交记录.md`](docs/提交记录.md)。

## 2. 关键发现：赛题数据与 GWHD 同源

用 **dHash 召回 + 像素 MAD 确认**（`scripts/check_overlap.py`）比对 1000 张赛题图与 3373 张 GWHD 图：

| 划分 | 总数 | 同图重编码（MAD<6/255） | 占比 |
|---|---|---|---|
| train | 800 | 434 | **54.2%** |
| val | 100 | 58 | **58.0%** |
| test | 100 | 47 | **47.0%** |

- 不是"相似领域"，是**同一批图的重编码版本**（像素差 1~2/255）；
- 但**标注口径不同**（同图：赛题 64 框 vs GWHD 53 框）⇒ 外部数据只适合**预训练**，最终必须回到赛题数据微调；
- **合规**：与 test 同源的 47 张（及 train/val 同源的 492 张）**全部剔除**，训练集不含任何测试图信息。见 `scripts/build_gwhd_clean.py` 与 `docs/重叠检查.md`。

## 3. 快速开始

```powershell
# 环境（本机 pip 不可用，见 docs/环境说明.md）
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\env.ps1
& $env:WHEAT_PY scripts\check_env.py            # 环境 + 数据自检

# 训练（示例：GWHD 预训练 → 900 微调，全流程）
.\run_long.ps1 -Deadline "08:00"

# 跨夜多模型 + 集成
.\run_overnight.ps1 -Deadline "08:40"

# 推理 / 集成 / 打包提交
& $env:WHEAT_PY scripts\predict.py --weights runs\detect\long_y11l_1024\weights\best.pt `
    --source data\dataset\images\test --out preds\test_l --imgsz 1024 --conf 0.001
& $env:WHEAT_PY scripts\predict_ensemble.py --weights a.pt b.pt `
    --source data\dataset\images\test --out preds\ens --imgsz 896 1024 1280 --fuse nms --nms-iou 0.6
& $env:WHEAT_PY scripts\make_submission.py --pred preds\ens --tag ens
```

## 4. 提交格式（官方要求）

`submit.zip` → `submit/` → 与测试图**同名的 txt**（UTF-8），每行：

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

归一化坐标、空格分隔、`class_id=0`。`make_submission.py` 会校验：字段数、坐标范围、class_id、文件名齐全、无 BOM、无 CRLF。

## 5. 仓库结构

```
├─ env.ps1                  # 环境激活（PYTHONPATH / 编码 / certifi / 便携 git）
├─ train.py  train.ps1      # 训练入口（--resume-from / --save-period / --fraction）
├─ run_long.ps1             # 长训练编排：预训练 → 微调 → 集成（显存自适应）
├─ run_overnight.ps1        # 跨夜：4 模型 → soup → 6 成员集成 → 2 个提交候选
├─ run_folds.ps1            # 5 折交叉验证
├─ sandbox_compat.py        # 沙箱兼容层（必须在 ultralytics 之前 import）
├─ data/
│  ├─ wheat.yaml            # 800/100 官方划分
│  ├─ wheat_full900.yaml    # 900 全量（train+val）
│  └─ folds/                # 5 折分层划分（按框密度）
├─ scripts/
│  ├─ check_env.py          # 环境 + 数据集自检
│  ├─ viz_labels.py         # 标注可视化
│  ├─ predict.py            # 单模型推理 → 提交格式 txt
│  ├─ predict_ensemble.py   # 多模型 × 多尺度 → NMS / WBF
│  ├─ eval_submission.py    # 用 txt 复算 mAP@0.5（独立校验导出）
│  ├─ oof_eval.py           # 5 折 OOF 评测
│  ├─ make_submission.py    # 校验 + 打包 submit.zip
│  ├─ make_folds.py         # 按密度分层切 5 折
│  ├─ model_soup.py         # 多 checkpoint 权重平均
│  ├─ make_experiment_log.py# 自动生成 docs/实验记录.md
│  ├─ check_overlap.py      # 赛题数据 vs GWHD 同源核查
│  ├─ build_gwhd_clean.py   # 剔除同源图，生成干净外部集
│  ├─ fetch_gwhd.py / fetch_gwhd_more.py  # 外部数据下载
│  ├─ prepare_gwhd.py       # parquet → YOLO
│  └─ add_pkg.py            # pip 替代安装器
└─ docs/
   ├─ 赛题要求.md           # 规则原文要点 + 落地校验清单
   ├─ 环境说明.md           # 为什么不能用 pip、三项沙箱限制
   ├─ 重叠检查.md           # 同源证据与合规筛查
   ├─ 实验记录.md           # 自动生成的实验表
   ├─ 提交记录.md           # 线上提交与分数
   └─ 冲刺96_三轮计划.md    # 冲 96 分的路线与止损线
```

## 6. 工程决策（踩过的坑）

| 问题 | 处理 |
|---|---|
| 本机 `pip install` 必失败（沙箱禁止临时目录删除/重命名） | 预解包包树 `pkgs/` + `PYTHONPATH`；新包用 `scripts/add_pkg.py` |
| `multiprocessing` 被禁（命名管道 → WinError 5） | `sandbox_compat.py` 把池换成串行，`workers=0` |
| 工作区路径含中文 | `.ps1` 一律用 `$PSScriptRoot`；`.ps1` 必须带 UTF-8 BOM（否则 GBK 解码报语法错）；`cv2.imread` 换 `np.fromfile + imdecode` |
| Python 默认 CA 缺中间证书 | `env.ps1` 设 `SSL_CERT_FILE` 指向 certifi |
| git 直连失败 | 走系统代理 + openssl 后端 + 把 CA 复制到 ASCII 路径（`env.ps1` 自动做） |
| 提交 txt 换行符 | 强制 LF，`make_submission.py` 兜底归一化 CRLF |
| 评测用完整 PR 曲线 | 推理 `conf=0.001`、`max_det=1000`；**不开翻转 TTA**（实测掉点） |

## 7. 数据与合规

- 赛题数据：800/100/100，全部 1024×1024，单类，标签 0 坏行
- 外部数据：GWHD（同源），剔除 539 张同源图后 **2834 张**用于预训练
- **测试集只用于推理**：不训练、不伪标签、不使用任何与测试图对应的外部标注

## 8. 环境

- RTX 4090D 24 GB / i9-14900KF / Windows / Python 3.12.14
- torch 2.6.0+cu124 + ultralytics 8.4.203（`pkgs/` 预解包，未入库）
- 详细说明：[`docs/环境说明.md`](docs/环境说明.md)
