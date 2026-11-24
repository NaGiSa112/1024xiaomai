# 跨夜长项目 v3：00:05 -> 09:50（用户 8 点起床上课，可跑到 10 点）
#
# 关键变化（v2 -> v3）：
#   1) 预训练数据换成 **gwhd_clean_v2（7770 张，8.6× 赛题 900 张）**
#      —— 已用像素 md5 证明赛题 1000 张全部来自 GWHD2021，且这 7770 张与赛题零重叠
#   2) 增加第 5 个成员：**l 的"加料版"**（同样吃 v2 预训练），给集成更多异构性
#   3) 窗口从 8.5h 拉到 9.7h
#
# 模型矩阵（时间按实测标定：l@1024 = 71 s/epoch/900 张；x@768 = 0.056 s/张）：
#   角色            模型     分辨率  数据        ep  bs  显存   预计耗时
#   预训练 A        yolo11x  768     v2-7770     12  8   17.5G  1.5 h
#   主力 A          yolo11x  1024    900         70  4   15.5G  3.1 h
#   主力 B（分辨率） yolo11l  1280    900         50  6   15.9G  1.5 h
#   主力 C（容量）   yolo11m  1024    900         60  10  14.6G  1.0 h
#   预训练 B        yolo11l  768     v2-7770     10  12  10.4G  1.0 h
#   主力 D（加料版） yolo11l  1024    900         40  8   13.8G  0.8 h
#   已有成员        yolo11l  1024    900         140 -   -      今天已训完
#   已有成员        fold0 s  1024    720         100 -   -      已完成
#
# 用法：
#   .\run_overnight.ps1 -Deadline "09:50"
#   .\run_overnight.ps1 -Deadline "09:50" -SkipD    # 不做加料版 l
param(
    [string]$Deadline = "09:50",
    [switch]$SkipWait,
    [switch]$SkipM,
    [switch]$SkipD,
    [int]$XPreEpochs = 12,
    [int]$XEpochs = 70,
    [int]$L1280Epochs = 50,
    [int]$MEpochs = 60,
    [int]$LPreEpochs = 10,
    [int]$LEpochs = 40
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Path (Join-Path $logs ("overnight_" + (Get-Date -Format "MMdd_HHmm") + ".log")) -Append | Out-Null

function Banner($t) { Write-Host "`n===== $t =====" -ForegroundColor Cyan }
$dl = [datetime]::Today + [timespan]::Parse($Deadline)
if ($dl -le (Get-Date)) { $dl = $dl.AddDays(1) }
function Remaining { [math]::Round(($dl - (Get-Date)).TotalHours, 2) }

# ---- 等到 00:01 再动 GPU
if (-not $SkipWait) {
    $start = [datetime]::Today.AddDays(1).AddMinutes(1)
    if ((Get-Date) -lt $start) {
        $waitMin = [math]::Round(($start - (Get-Date)).TotalMinutes, 0)
        Write-Host "[wait] 现在 $((Get-Date).ToString('HH:mm'))，等 $waitMin 分钟到 00:01 再启动"
        Start-Sleep -Seconds ([int](($start - (Get-Date)).TotalSeconds))
    }
}
Write-Host "[overnight v3] start $((Get-Date).ToString('HH:mm'))  deadline $dl  剩余 $(Remaining) h"

function Train($name, $model, $data, $epochs, $imgsz, $batch, $extra) {
    $best = "runs\detect\$name\weights\best.pt"
    $last = "runs\detect\$name\weights\last.pt"
    if (Test-Path $best) { Write-Host "[skip] $name 已完成"; return }
    if (Test-Path $last) {
        Write-Host "[resume] $name <- last.pt"
        & $env:WHEAT_PY train.py --resume-from $last
        return
    }
    Write-Host "[train] $name : $model @$imgsz x$epochs bs$batch"
    & $env:WHEAT_PY train.py --model $model --data $data --epochs $epochs --imgsz $imgsz `
        --batch $batch --cache ram --name $name @extra
}

$V2 = "data_external/gwhd_clean_v2.yaml"

# ---------------------------------------------------------------- 1 x 预训练（v2 数据）
Banner "1/6 yolo11x 预训练 @768 on GWHD-v2(7770) x$XPreEpochs"
if (Test-Path "runs\detect\x_pre_v2_768\weights\best.pt") {
    Write-Host "[skip] x 预训练已完成"
}
elseif (Test-Path "runs\detect\x_pre_v2_768\weights\last.pt") {
    & $env:WHEAT_PY train.py --resume-from "runs\detect\x_pre_v2_768\weights\last.pt"
}
else {
    & $env:WHEAT_PY train.py --model "runs/detect/x_pre_gwhd_768/weights/best.pt" --data $V2 `
        --epochs $XPreEpochs --imgsz 768 --batch 8 --cache ram `
        --close-mosaic 4 --patience 20 --seed 1 --name x_pre_v2_768
}

# ---------------------------------------------------------------- 2 x 微调
Banner "2/6 yolo11x 微调 @1024 on 900 x$XEpochs"
$xInit = if (Test-Path "runs\detect\x_pre_v2_768\weights\best.pt") { "runs/detect/x_pre_v2_768/weights/best.pt" } else { "runs/detect/x_pre_gwhd_768/weights/best.pt" }
Train "x900_1024" $xInit "data/wheat_full900.yaml" $XEpochs 1024 4 @("--close-mosaic", "10", "--patience", "30", "--seed", "3", "--save-period", "5")

# ---------------------------------------------------------------- 3 l @1280
if ((Remaining) -gt 3.2) {
    Banner "3/6 yolo11l 微调 @1280 on 900 x$L1280Epochs"
    $lInit = if (Test-Path "runs\detect\long_y11l_1024\weights\best.pt") { "runs/detect/long_y11l_1024/weights/best.pt" } else { "weights/yolo11l.pt" }
    Train "l1280_900" $lInit "data/wheat_full900.yaml" $L1280Epochs 1280 6 @("--close-mosaic", "10", "--patience", "25", "--seed", "5")
}
else { Write-Host "[skip] 剩余 $(Remaining) h 不足 3.2h，跳过 l@1280" }

# ---------------------------------------------------------------- 4 m @1024
if (-not $SkipM -and (Remaining) -gt 2.2) {
    Banner "4/6 yolo11m 微调 @1024 on 900 x$MEpochs"
    Train "m900_1024" "weights/yolo11m.pt" "data/wheat_full900.yaml" $MEpochs 1024 10 @("--close-mosaic", "10", "--patience", "25", "--seed", "7")
}
else { Write-Host "[skip] m 模型" }

# ---------------------------------------------------------------- 5 l 加料版（v2 预训练）
if (-not $SkipD -and (Remaining) -gt 2.0) {
    Banner "5/6 yolo11l 加料版：预训练 @768 on v2 x$LPreEpochs -> 微调 @1024 x$LEpochs"
    Train "l_pre_v2_768" "weights/yolo11l.pt" $V2 $LPreEpochs 768 12 @("--close-mosaic", "4", "--patience", "15", "--seed", "11")
    if (Test-Path "runs\detect\l_pre_v2_768\weights\best.pt") {
        Train "l_v2_900" "runs/detect/l_pre_v2_768/weights/best.pt" "data/wheat_full900.yaml" $LEpochs 1024 8 @("--close-mosaic", "8", "--patience", "20", "--seed", "13")
    }
}
else { Write-Host "[skip] 加料版 l" }

# ---------------------------------------------------------------- 6 soup + 集成
Banner "6/6 soup + 集成"
$epochs = Get-ChildItem "runs\detect\x900_1024\weights" -Filter "epoch*.pt" -ErrorAction SilentlyContinue | Sort-Object Name
if ($epochs.Count -ge 2) {
    $paths = @("runs\detect\x900_1024\weights\best.pt") + ($epochs | Select-Object -Last 4 | ForEach-Object { $_.FullName })
    & $env:WHEAT_PY scripts\model_soup.py --weights @paths --out "weights/x_soup.pt"
}

$weights = @()
foreach ($n in @("long_y11l_1024", "x900_1024", "l1280_900", "m900_1024", "l_v2_900", "fold0_y11s_1024")) {
    if ($n -and (Test-Path "runs\detect\$n\weights\best.pt")) { $weights += "runs\detect\$n\weights\best.pt" }
}
if (Test-Path "weights\x_soup.pt") { $weights += "weights/x_soup.pt" }
Write-Host "[ens] $($weights.Count) 个成员"
$weights | ForEach-Object { Write-Host "   - $_" }

if ($weights.Count -gt 0) {
    $stamp = Get-Date -Format "MMdd"
    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_on_${stamp}_nms" `
        --imgsz 896 1024 1280 --fuse nms --nms-iou 0.6
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_on_${stamp}_nms" --tag "on_${stamp}_nms"

    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_on_${stamp}_solo" `
        --imgsz 1024 --fuse nms --nms-iou 0.6
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_on_${stamp}_solo" --tag "on_${stamp}_solo"
}

Banner "完成"
Get-ChildItem submissions\*.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 3 Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 2) } }
Stop-Transcript | Out-Null
