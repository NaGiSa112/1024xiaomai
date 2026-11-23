# 小麦麦穗目标检测（Wheat-SOD）· 讯飞算法赛

单类别目标检测：田间实拍 **1024×1024** 小麦图像，识别并定位麦穗（`wheat_ear`）。
赛题官方指标 **mAP@0.5**。

- 赛题：<https://challenge.xfyun.cn/topic/info?type=Wheat-SOD-train>
- 数据：train 800 / val 100 / test 100（test 无标注，仅推理）
- 目标：密集**中小目标** —— 平均 42.7 框/图，框宽中位数 0.068（1024 图上约 68 px）

---

## 1. 结果

| 实验 | 模型 | imgsz | 本地 mAP50 | mAP50-95 | 说明 |
|---|---|---|---|---|---|
| smoke @640 | yolo11n | 640 | 0.750 | 0.362 | 2 epoch，链路验证 |
| smoke @1024 | yolo11s | 1024 | 0.862 | 0.433 | 3 epoch，76 s/epoch |
| baseline | yolo11l | 1280 | _待填_ | _待填_ | 150 epoch |
| final | 跨架构集成 | 多尺度 | _待填_ | _待填_ | 线上分数见下 |

> 线上排行榜分数与提交记录见 `docs/冲刺95方案.md`。

## 2. 快速开始

```powershell
# 0) 环境（本机 pip 不可用，见 docs/环境说明.md）
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\env.ps1
& $env:WHEAT_PY scripts\check_env.py          # 环境 + 数据集自检

# 1) 训练
.\train.ps1 "train.py","--model","weights/yolo11s.pt","--epochs","150","--imgsz","1280","--batch","8","--name","y11s_1280"

# 2) 推理（val 用于确认导出无误；test 用于提交）
& $env:WHEAT_PY scripts\predict.py --weights runs\detect\y11s_1280\weights\best.pt `
    --source data\dataset\images\val --out preds\val_y11s_1280 --imgsz 1280 --conf 0.001

# 3) 打包提交（自动校验格式）
& $env:WHEAT_PY scripts\make_submission.py --pred preds\test_y11s_1280 --tag y11s_1280
```

## 3. 提交格式（官方要求）

`submit.zip` → `submit/` → 与测试图**同名的 txt**（UTF-8），每行：

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

- 坐标为**归一化**值（相对宽高，[0,1]），空格分隔
- `class_id` 从 0 开始（本赛题仅 0 = 麦穗）
- 每队每天最多 3 次提交，取历史最优

## 4. 目录结构

```
.
├─ env.ps1                  # 环境激活（PYTHONPATH / 编码 / certifi / 便携 git）
├─ train.py  train.ps1      # 训练入口（分组参数、summary.json）
├─ sandbox_compat.py        # 沙箱兼容层（必须在 ultralytics 之前 import）
├─ data/
│  ├─ wheat.yaml            # 800/100 干净划分
│  └─ wheat_full900.yaml    # 900 全量（train+val）
├─ scripts/
│  ├─ check_env.py          # 环境 + 数据集自检
│  ├─ viz_labels.py         # 标注可视化
│  ├─ predict.py            # 单模型推理 -> 提交格式 txt
│  ├─ predict_ensemble.py   # 多模型 × 多尺度 -> NMS / WBF 融合
│  ├─ eval_submission.py    # 用 txt 复算 mAP@0.5（独立校验）
│  ├─ make_submission.py    # 校验 + 打 zip
│  ├─ add_pkg.py            # pip 替代安装器（下载 wheel 解包）
│  ├─ fetch_gwhd.py         # GWHD 外部数据获取（逐个文件版）
│  └─ prepare_gwhd.py       # GWHD parquet -> YOLO 数据集
├─ docs/
│  ├─ 赛题要求.md
│  ├─ 环境说明.md
│  └─ 冲刺95方案.md
└─ runs/  preds/  submissions/   (不入库)
```

## 5. 关键工程决策

| 问题 | 处理 |
|---|---|
| 本机 `pip install` 失败（沙箱禁止临时目录删除/重命名） | 预解包包树 `pkgs/` + `PYTHONPATH`；新包用 `scripts/add_pkg.py` |
| `multiprocessing` 被禁（命名管道） | `sandbox_compat.py` 把池换成串行，`workers=0` |
| 工作区路径含中文 | `.ps1` 全部用 `$PSScriptRoot` 拼路径；`cv2.imread/imwrite` 换成 `fromfile/imdecode` |
| Python 默认 CA 缺中间证书 | `env.ps1` 里设 `SSL_CERT_FILE` 指向 certifi |
| 提交 txt 的换行符 | 强制 LF，`make_submission.py` 兜底把 CRLF 归一化 |
| 评测用完整 PR 曲线 | 推理 `conf=0.001`，`max_det=1000`，不开翻转 TTA（实测掉点） |

## 6. 数据

| 划分 | 图片 | 框数 | 平均框/图 |
|---|---|---|---|
| train | 800 | 34,166 | 42.7 |
| val | 100 | 4,675 | 46.8 |
| test | 100 | — | — |

外部数据（经许可使用）：**GWHD**（Global Wheat Head Detection）3,373 张 1024×1024、147,791 框，
标注格式与赛题一致（同源领域，密度 43.8 框/图 ≈ 本赛题 42.7）。
