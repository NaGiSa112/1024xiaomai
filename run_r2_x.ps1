# R2：yolo11x（最强骨干）—— GWHD-clean 预训练 15ep@768 -> 900 微调 60ep@1024 -> 集成出 zip
#
# 显存依据（实测 yolo11l@1024 bs8 = 13.8 GB -> 单图 1.54 GB）：
#   激活显存 ≈ FLOPs 比：x/l = 56.9/25.3 = 2.25
#   x@1024 单图 ≈ 3.5 GB -> bs4 约 15.4 GB（bs6 会到 22 GB，太贴边）
#   x@768  单图 ≈ 2.0 GB -> bs8 约 17.1 GB
#
# 时间依据（实测 l@1024 = 73 s/epoch，单图 0.081 s）：
#   x@768  单图 0.102 s -> 2834 张 = 290 s/epoch -> 15 ep ≈ 1.2 h
#   x@1024 单图 0.182 s -> 900 张  = 164 s/epoch -> 60 ep ≈ 2.7 h
param(
    [string]$Tag = "x900_1024",
    [int]$PreEpochs = 15,
    [int]$FineEpochs = 60,
    [int]$FineBatch = 4,
    [switch]$SkipEnsemble
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Path (Join-Path $logs ("r2x_" + (Get-Date -Format "MMdd_HHmm") + ".log")) -Append | Out-Null

function Banner($t) { Write-Host "`n===== $t =====" -ForegroundColor Cyan }
function Train($name, $model, $data, $epochs, $imgsz, $batch, $extra) {
    if (Test-Path "runs\detect\$name\weights\best.pt") { Write-Host "[skip] $name"; return }
    if (Test-Path "runs\detect\$name\weights\last.pt") {
        Write-Host "[resume] $name"
        & $env:WHEAT_PY train.py --resume-from "runs\detect\$name\weights\last.pt"
        return
    }
    Write-Host "[train] $name ($model @$imgsz x$epochs, bs$batch)"
    & $env:WHEAT_PY train.py --model $model --data $data --epochs $epochs --imgsz $imgsz `
        --batch $batch --cache ram --name $name @extra
}

Banner "R2-1 yolo11x 预训练 @768 on GWHD-clean ($PreEpochs ep)"
Train "x_pre_gwhd_768" "weights/yolo11x.pt" "data_external/gwhd_clean.yaml" $PreEpochs 768 8 @("--close-mosaic", "5", "--patience", "20", "--seed", "0")

Banner "R2-2 yolo11x 微调 @1024 on 900 ($FineEpochs ep)"
if (Test-Path "runs\detect\x_pre_gwhd_768\weights\best.pt") {
    Train $Tag "runs/detect/x_pre_gwhd_768/weights/best.pt" "data/wheat_full900.yaml" $FineEpochs 1024 $FineBatch @("--close-mosaic", "10", "--patience", "25", "--seed", "3")
} else {
    Write-Host "[warn] 预训练没有产出，直接用 COCO 权重微调"
    Train $Tag "weights/yolo11x.pt" "data/wheat_full900.yaml" $FineEpochs 1024 $FineBatch @("--close-mosaic", "10", "--patience", "25", "--seed", "3")
}

if ($SkipEnsemble) { Stop-Transcript | Out-Null; return }

Banner "R2-3 集成：l + x + fold0 多尺度 -> submit.zip"
$weights = @()
foreach ($n in @("long_y11l_1024", $Tag, "long_y11m_1024", "fold0_y11s_1024")) {
    if ($n -and (Test-Path "runs\detect\$n\weights\best.pt")) { $weights += "runs\detect\$n\weights\best.pt" }
}
Write-Host "[ens] $($weights.Count) 个模型: $($weights -join ', ')"
if ($weights.Count -gt 0) {
    $sub = "R2_" + (Get-Date -Format "MMdd_HHmm")
    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_$sub" `
        --imgsz 896 1024 1280 --fuse nms --nms-iou 0.6
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_$sub" --tag $sub
    Write-Host "[done] submissions\submit_$sub.zip"
}

Stop-Transcript | Out-Null
