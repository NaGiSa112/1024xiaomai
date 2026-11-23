# Build a submission from the 5 fold models: multi-scale + NMS fusion.
# Usage: .\submit_ensemble.ps1 -Tag ens5_ms -Scales 896 1024 1280 -NmsIou 0.6
param(
    [string]$Tag = "ens5_ms",
    [int[]]$Scales = @(896, 1024, 1280),
    [double]$NmsIou = 0.6,
    [string]$Fuse = "nms",
    [double]$BoxScale = 1.0,
    [string]$Source = "data/dataset/images/test"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$weights = @()
foreach ($f in 0..4) {
    $w = "runs\detect\fold${f}_y11s_1024\weights\best.pt"
    if (Test-Path $w) { $weights += $w } else { Write-Host "[warn] missing $w" }
}
if ($weights.Count -eq 0) { throw "没有可用的折模型，先用 run_folds.ps1 训练" }

Write-Host "[ens] $($weights.Count) 个模型 x 尺度 $($Scales -join ',') -> preds/test_$Tag"
& $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
    --source $Source --out "preds/test_$Tag" `
    --imgsz @Scales --fuse $Fuse --nms-iou $NmsIou --box-scale $BoxScale
if ($LASTEXITCODE -ne 0) { throw "predict_ensemble failed" }

& $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_$Tag" --tag $Tag
if ($LASTEXITCODE -ne 0) { throw "make_submission failed (格式校验未通过)" }

Write-Host "[done] submissions\submit_$Tag.zip"
