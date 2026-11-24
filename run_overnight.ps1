# 跨夜长项目（00:05 -> 08:40）：训练 4 个强模型 -> 5 成员多尺度集成 -> 2 个提交候选
#
# 前提：17:00-24:00 用户自用，GPU 完全不动；00:00 之后才允许启动（脚本自带等待）
#
# 模型矩阵（显存按实测 l@1024 bs8 = 13.8 GB / 单图 1.54 GB 推算）：
#   角色             模型            分辨率  数据        epoch  bs   显存估  实测/推算 epoch 时间
#   预训练           yolo11x         768     GWHD-2834   15     8    17.5GB  293 s
#   主力 A           yolo11x         1024    900         75     4    15.5GB  165 s
#   主力 B（分辨率）  yolo11l         1280    900         55     6    15.9GB  115 s
#   主力 C（容量）    yolo11m         1024    900         70     10   14.6GB  62 s
#   已有成员         yolo11l         1024    900         140    -    -       73.5 s（今天已训完）
#   已有成员         yolo11s(fold0)  1024    720         100    -    -       -
#
# 用法：
#   .\run_overnight.ps1 -Deadline "08:40"
#   .\run_overnight.ps1 -Deadline "08:40" -SkipM          # 砍掉 m，只留 x + l1280
param(
    [string]$Deadline = "08:40",
    [switch]$SkipM,
    [switch]$SkipWait,
    [int]$XEpochs = 75,
    [int]$L1280Epochs = 55,
    [int]$MEpochs = 70
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

# ---- 等到 00:01 再动 GPU（避免占用用户 17:00-24:00 的机器时间）
if (-not $SkipWait) {
    $start = [datetime]::Today.AddDays(1).AddMinutes(1)     # 次日 00:01
    if ((Get-Date) -lt $start) {
        $waitMin = [math]::Round(($start - (Get-Date)).TotalMinutes, 0)
        Write-Host "[wait] 现在 $((Get-Date).ToString('HH:mm'))，等 $waitMin 分钟到 00:01 再启动（不占用你晚上的机器）"
        Start-Sleep -Seconds ([int](($start - (Get-Date)).TotalSeconds))
    }
}
Write-Host "[overnight] start $((Get-Date).ToString('HH:mm'))  deadline $dl  剩余 $(Remaining) h"

function Train($name, $model, $data, $epochs, $imgsz, $batch, $extra) {
    if (Test-Path "runs\detect\$name\weights\best.pt") { Write-Host "[skip] $name 已完成"; return }
    if (Test-Path "runs\detect\$name\weights\last.pt") {
        Write-Host "[resume] $name"
        & $env:WHEAT_PY train.py --resume-from "runs\detect\$name\weights\last.pt"
        return
    }
    Write-Host "[train] $name  $model @$imgsz x$epochs bs$batch"
    & $env:WHEAT_PY train.py --model $model --data $data --epochs $epochs --imgsz $imgsz `
        --batch $batch --cache ram --name $name @extra
}

# ---------------------------------------------------------------- 1 yolo11x 预训练
Banner "1/4 yolo11x 预训练 @768 on GWHD-clean(2834) x15"
Train "x_pre_gwhd_768" "weights/yolo11x.pt" "data_external/gwhd_clean.yaml" 15 768 8 @("--close-mosaic", "5", "--patience", "20", "--seed", "0")

# ---------------------------------------------------------------- 2 yolo11x 微调
Banner "2/4 yolo11x 微调 @1024 on 900 x$XEpochs"
$xInit = if (Test-Path "runs\detect\x_pre_gwhd_768\weights\best.pt") { "runs/detect/x_pre_gwhd_768/weights/best.pt" } else { "weights/yolo11x.pt" }
Train "x900_1024" $xInit "data/wheat_full900.yaml" $XEpochs 1024 4 @("--close-mosaic", "10", "--patience", "30", "--seed", "3", "--save-period", "5")

# ---------------------------------------------------------------- 3 yolo11l @1280（分辨率维度）
if ((Remaining) -gt 2.2) {
    Banner "3/4 yolo11l 微调 @1280 on 900 x$L1280Epochs（从今天 1024 模型热启动）"
    $lInit = if (Test-Path "runs\detect\long_y11l_1024\weights\best.pt") { "runs/detect/long_y11l_1024/weights/best.pt" } else { "weights/yolo11l.pt" }
    Train "l1280_900" $lInit "data/wheat_full900.yaml" $L1280Epochs 1280 6 @("--close-mosaic", "10", "--patience", "25", "--seed", "5")
}
else { Write-Host "[skip] 剩余 $(Remaining) h 不足，跳过 l@1280" }

# ---------------------------------------------------------------- 4 yolo11m @1024（容量维度）
if (-not $SkipM -and (Remaining) -gt 1.8) {
    Banner "4/4 yolo11m 微调 @1024 on 900 x$MEpochs"
    Train "m900_1024" "weights/yolo11m.pt" "data/wheat_full900.yaml" $MEpochs 1024 10 @("--close-mosaic", "10", "--patience", "25", "--seed", "7")
}
else { Write-Host "[skip] m 模型（-SkipM 或时间不足）" }

# ---------------------------------------------------------------- soup（x 的最后几个 epoch）
Banner "补齐：x 的 model soup"
$epochs = Get-ChildItem "runs\detect\x900_1024\weights" -Filter "epoch*.pt" -ErrorAction SilentlyContinue | Sort-Object Name
if ($epochs.Count -ge 2) {
    $last4 = $epochs | Select-Object -Last 4
    $paths = @("runs\detect\x900_1024\weights\best.pt") + ($last4 | ForEach-Object { $_.FullName })
    & $env:WHEAT_PY scripts\model_soup.py --weights @paths --out "weights/x_soup.pt"
}
else { Write-Host "[skip] 没有 epoch*.pt（--save-period 没生效或训练未完成）" }

# ---------------------------------------------------------------- 集成 + 两个提交候选
Banner "集成：全部成员 x 多尺度 -> 2 个 zip"
$weights = @()
foreach ($n in @("long_y11l_1024", "x900_1024", "l1280_900", "m900_1024", "fold0_y11s_1024")) {
    if ($n -and (Test-Path "runs\detect\$n\weights\best.pt")) { $weights += "runs\detect\$n\weights\best.pt" }
}
if (Test-Path "weights\x_soup.pt") { $weights += "weights\x_soup.pt" }
Write-Host "[ens] $($weights.Count) 个成员：`n  $($weights -join "`n  ")"

if ($weights.Count -gt 0) {
    $stamp = Get-Date -Format "MMdd"
    # 候选 1：NMS 0.6
    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_on_${stamp}_nms" `
        --imgsz 896 1024 1280 --fuse nms --nms-iou 0.6
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_on_${stamp}_nms" --tag "on_${stamp}_nms"
    # 候选 2：WBF 0.55（做 A/B）
    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_on_${stamp}_wbf" `
        --imgsz 896 1024 1280 --fuse wbf --wbf-iou 0.55
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_on_${stamp}_wbf" --tag "on_${stamp}_wbf"
}

Banner "完成"
Get-ChildItem submissions\*.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 3 Name, @{n="MB"; e={[math]::Round($_.Length / 1MB, 2)}}
Stop-Transcript | Out-Null
