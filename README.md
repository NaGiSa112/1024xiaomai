# 小麦麦穗目标检测 · 赛题方案与工程实现

> 讯飞 AI 开发者大赛「小麦麦穗目标检测挑战赛」的完整实现：数据核查 → 外部数据构建 → 预训练 → 微调 → 多模型集成 → 提交。
> 线上 mAP@0.5：**0.92187**（接手时前任版本为 0.91548）

---

## 一、赛题简介

| 项 | 内容 |
|---|---|
| 任务 | 单类别目标检测，类别 `wheat_ear`（麦穗） |
| 数据 | 田间实拍图像（本赛题全部为 1024×1024），**训练集、验证集带完整标注，测试集仅提供图像** |
| 标注格式 | YOLO：`<class_id> <x_center> <y_center> <width> <height>`，归一化坐标，类别编号从 0 开始 |
| 评测指标 | **mAP@0.5**（依据提交的检测结果文件计算，按分数从高到低排名） |
| 提交要求 | `submit.zip` 内含 `submit/` 文件夹，文件夹内为**与测试图同名的 txt**（UTF-8），每行：`<class_id> <x_center> <y_center> <width> <height> <confidence>`，空格分隔、归一化坐标 |
| 提交限制 | 每支队伍**每天最多 3 次** |
| 关键约束 | **测试集仅可用于输出最终推理结果，不可以任何形式参与训练过程** |

**数据特点（本地实测）**

| 划分 | 图片 | 标注框 | 平均框/图 | 说明 |
|---|---|---|---|---|
| train | 800 | 34,166 | 42.7 | 单图 1~190 个目标 |
| val | 100 | 4,675 | 46.8 | 官方验证集 |
| test | 100 | — | — | 仅图像 |

- 目标密集且偏小：框宽中位数 **0.068**（1024 图上约 68 px），`sqrt(area)` 中位数同为 68 px；
- 因此**输入分辨率**是关键变量（640→1024 单此一项就有 +1.1 点），推理也必须保留低置信度候选（`conf=0.001`、`max_det=1000`）以走完整 PR 曲线。

## 二、我们的工作

```
赛题数据 900 张 ─┐
                ├─► 预训练（GWHD-clean 7770 张）─► 微调（900 张全量）─► 多模型多尺度集成 ─► submit.zip
外部 GWHD 数据 ──┘        同源、去重、剔除测试集重叠        对齐官方标注口径
```

### 2.1 关键发现：赛题数据与公开数据集 GWHD 同源

用 **dHash 召回 + 像素级 MAD 确认**，随后用 GWHD2021 官方全量包做**像素 md5 精确核对**：

| 项 | 结果 |
|---|---|
| 赛题 1000 张在 GWHD2021 中的命中 | **1000 / 1000（100%）**，包含全部 100 张测试图 |
| 判定口径 | 解码成 numpy 后取 md5 —— 重编码不改像素，假阳性为零 |
| 早期教训 | 只用 dHash + MAD<6，在 6515×3373 规模上假阳性爆炸（5515 张全判重复），换成像素 md5 才正确 |

这条发现直接决定策略：外部数据不是"相似领域"，而是**同分布扩增**；同时因为同源，**必须**把与测试图相同的样本剔除出去。

### 2.2 数据构建（收益最大的一环）

| 步骤 | 内容 | 结果 |
|---|---|---|
| 外部数据获取 | HF 镜像（GWHD2020 子集）+ ModelScope 全量包 | 9.5 GB / 6515 张 |
| 合规剔除 | 像素 md5 比对，剔除与赛题 1000 张重复的图 | 剔除 1000 张（含全部测试图同源样本） |
| 去重 | 再剔除与已有外部集重复的 676 张 | — |
| **干净外部集** | 合并后 | **7770 张（赛题训练集的 8.6 倍）** |
| 标注转换 | GWHD CSV（像素 `x1 y1 x2 y2`）→ YOLO 归一化 | 275,458 个框 |

**数据规模的实测影响（干净口径：模型从未见过任何赛题图）**

| 预训练数据 | 模型 | 干净 val mAP@0.5 |
|---|---|---|
| 2834 张 | yolo11l @768，40 ep | 0.8812 |
| **7770 张** | yolo11x @768，12 ep | **0.9449** |

→ 数据量 2.7 倍，**干净指标 +6.4 点**，比换骨干/改结构的收益大得多。

### 2.3 模型与推理

| 环节 | 做法 |
|---|---|
| 预训练 | GWHD-clean @768，12~40 epoch（同源特征） |
| 微调 | 赛题 900 张全量 `@1024` 或 `@1280`，`close_mosaic` 末期关闭，cos_lr |
| 骨干 | yolo11 s / m / l / x 四档（集成需要异构，同族集成会饱和） |
| 分辨率 | 1024（原生）与 1280（`mAP50-95` 明显更高） |
| 加速 | `cache=ram` 训练提速 42%（65 s/ep → 38 s/ep） |
| 集成 | 多成员 × 多尺度（896/1024/1280）→ 加权 WBF / NMS |
| 校准 | `box_scale` 微调、`max_det=1000`、`conf=0.001` |
| 排除 | 翻转 TTA（实测掉点）、同族无限集成（饱和） |

## 三、结果

### 3.1 关键模型（指标取 best epoch）

| 模型 | epoch | mAP50 | mAP50-95 | P | R | 数据口径 |
|---|---|---|---|---|---|---|
| yolo11l @1280 | 50 | **0.9901** | **0.7325** | 0.966 | 0.971 | 900（val 参与训练） |
| yolo11l @1024 | 140 | 0.9887 | 0.7076 | 0.972 | 0.957 | 900（val 参与训练） |
| yolo11x @1024 | 80 | 0.9851 | 0.6757 | 0.960 | 0.958 | 900（val 参与训练） |
| yolo11m @1024 | 60 | 0.9716 | 0.6141 | 0.947 | 0.933 | 900（val 参与训练） |
| yolo11x 预训练 | 12 | **0.9449** | 0.5591 | 0.916 | 0.892 | **干净** |
| yolo11l 预训练 | 10 | **0.9380** | 0.5485 | 0.924 | 0.877 | **干净** |
| yolo11s（5 折第 2 折） | 100 | 0.9306 | 0.5359 | 0.908 | 0.880 | **干净折外** |
| yolo11s（5 折第 1 折） | 100 | 0.9108 | 0.5135 | 0.913 | 0.851 | **干净折外** |

> 「val 参与训练」= 用 900 张全量（含官方 val）训练，本地指标偏乐观，仅反映收敛；
> 「干净」= 模型完全没见过赛题数据，是真实泛化能力的证据。

### 3.2 线上提交

| # | 内容 | 线上 mAP@0.5 |
|---|---|---|
| 1 | 2 成员（l@1024 + fold0-s）× 896/1024/1280 + NMS | **0.92187** |
| 2 | 8 成员（跨 4 骨干 × 2 分辨率）× 多尺度 + NMS | 见 `docs/提交记录.md` |
| 3 | 大集成（5 折 OOF 选参后的最优融合） | 同上 |

## 四、快速开始

```powershell
# 环境（本机 pip 不可用，见 docs/环境说明.md）
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\env.ps1
& $env:WHEAT_PY scripts\check_env.py            # 环境 + 数据自检

# 外部数据：下载 → 转 YOLO → 合规剔除 → 干净集
& $env:WHEAT_PY scripts\fetch_ms_single.py
& $env:WHEAT_PY scripts\prepare_gwhd2021.py --extract
& $env:WHEAT_PY scripts\dedupe_exact.py

# 训练：预训练 → 微调 → 集成（显存自适应 batch）
.\run_overnight.ps1 -Deadline "11:00"

# 5 折补齐 + OOF 调参 + 大集成
.\run_extend.ps1 -Deadline "12:45"

# 推理与打包（自带格式校验：字段数/坐标范围/class_id/文件名齐全/无 BOM 无 CRLF）
& $env:WHEAT_PY scripts\predict.py --weights runs\detect\l1280_900\weights\best.pt `
    --source data\dataset\images\test --out preds\test_l1280 --imgsz 1280 --conf 0.001
& $env:WHEAT_PY scripts\make_submission.py --pred preds\test_l1280 --tag l1280
```

## 五、仓库结构

```
├─ train.py / train.ps1           训练入口（--resume-from / --save-period / --fraction）
├─ run_overnight.ps1              跨夜编排：预训练 → 微调 → soup → 集成
├─ run_extend.ps1                 续跑：补 5 折 → 加料版 → OOF 调参 → 大集成
├─ run_big_ensemble.ps1           大集成流水线（OOF 选参 → 全成员跑 test）
├─ sandbox_compat.py              受限环境兼容层（必须在 ultralytics 之前 import）
├─ data/
│  ├─ wheat.yaml                  官方 800/100 划分
│  ├─ wheat_full900.yaml          900 张全量
│  └─ folds/                      按目标密度分层的 5 折
├─ scripts/
│  ├─ check_env.py                环境 + 数据自检
│  ├─ check_overlap.py            与 GWHD 同源核查（dHash + 像素 MAD）
│  ├─ dedupe_exact.py             像素 md5 精确去重（合规剔除）
│  ├─ build_gwhd_clean.py         生成干净外部集
│  ├─ prepare_gwhd2021.py         GWHD2021 解压 + 转 YOLO
│  ├─ fetch_ms_single.py          ModelScope 单流下载
│  ├─ predict.py                  单模型推理 → 提交格式
│  ├─ ensemble_big.py             多成员 × 多尺度 → 加权 WBF / NMS
│  ├─ predict_ensemble.py         轻量版集成（少量成员）
│  ├─ tune_fusion_oof.py          用 5 折 OOF 搜融合参数
│  ├─ eval_submission.py          用导出 txt 复算 mAP@0.5
│  ├─ oof_eval.py                 5 折 OOF 评测
│  ├─ make_submission.py          格式校验 + 打包 submit.zip
│  ├─ make_folds.py               密度分层切 5 折
│  ├─ model_soup.py               多 checkpoint 权重平均
│  ├─ make_experiment_log.py      自动生成实验记录
│  └─ add_pkg.py                  pip 替代安装器
└─ docs/
   ├─ 项目总结.md                 完整项目总结（推荐先看）
   ├─ 赛题要求.md                 规则原文要点 + 落地清单
   ├─ 重叠检查.md                 同源证据与合规筛查
   ├─ 实验记录.md                 自动生成的实验表
   ├─ 提交记录.md                 线上提交与分数
   ├─ 环境说明.md                 环境搭建与限制
   ├─ 工程笔记.md                 踩坑与设计取舍
   └─ 冲刺96_三轮计划.md          目标路线与止损线
```

## 六、环境

| 项 | 版本 |
|---|---|
| GPU | RTX 4090D 24 GB |
| Python | 3.12.14 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| Ultralytics | 8.4.203 |

> 本项目环境不走 `pip`（宿主沙箱会拦截 pip 的临时目录重命名），依赖以预解包包树 + `PYTHONPATH` 挂载；
> 多进程被禁（`sandbox_compat.py` 接管）、路径含中文（脚本统一用 `$PSScriptRoot` 且 `.ps1` 强制 UTF-8 BOM）。
> 细节见 [`docs/环境说明.md`](docs/环境说明.md) 与 [`docs/工程笔记.md`](docs/工程笔记.md)。

## 七、合规说明

- 外部数据来自公开数据集 **GWHD**，与赛题数据同源；
- 用**像素级 md5** 把与赛题 1000 张（含全部测试图）重复的样本**全部剔除**，训练集零重叠；
- **未**使用 GWHD 中与测试图对应的标注、**未**做测试集伪标签、**未**在测试集上调参；
- 证据：`docs/重叠检查.md`、`data_external/gwhd2021_dedupe_summary.json`、`scripts/dedupe_exact.py`。

## 八、参考

- 赛题：<https://challenge.xfyun.cn/topic/info?type=Wheat-SOD-train>
- GWHD（Global Wheat Head Detection）：<https://www.globalwheatproject.org/>
