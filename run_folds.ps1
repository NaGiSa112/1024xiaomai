# 5-fold CV: train each fold, then predict its out-of-fold images.
# Usage: .\run_folds.ps1 -Epochs 100 -Batch 12 -Tag y11s_1024
param(
    [string]$Model = "weights/yolo11s.pt",
    [int]$Epochs = 100,
    [int]$Batch = 12,
    [string]$Imgsz = "1024",
    [string]$Tag = "y11s_1024",
    [int[]]$Folds = @(0, 1, 2, 3, 4)
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

foreach ($f in $Folds) {
    $name = "fold${f}_$Tag"
    $best = "runs\detect\$name\weights\best.pt"

    if (-not (Test-Path $best)) {
        Write-Host "=== [fold $f] train $name (epochs=$Epochs imgsz=$Imgsz batch=$Batch) ==="
        & $env:WHEAT_PY train.py --model $Model --data "data/folds/fold$f.yaml" `
            --epochs $Epochs --imgsz $Imgsz --batch $Batch --cache ram `
            --close-mosaic 15 --patience 30 --name $name
        if ($LASTEXITCODE -ne 0) { Write-Host "!!! fold $f training failed (exit $LASTEXITCODE)"; exit 1 }
    }
    else {
        Write-Host "=== [fold $f] skip training (best.pt exists) ==="
    }

    Write-Host "=== [fold $f] predict out-of-fold images ==="
    & $env:WHEAT_PY scripts\predict.py --weights $best `
        --source-list "data/folds/fold${f}_val.txt" --out "preds/oof_fold$f" `
        --imgsz $Imgsz --conf 0.001 --iou 0.6 --batch 8
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! fold $f predict failed"; exit 1 }
}

Write-Host "=== OOF evaluation ==="
& $env:WHEAT_PY scripts\oof_eval.py --folds $Folds
Write-Host "=== all folds done: $Tag ==="
