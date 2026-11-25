# 大集成流水线：补齐 5 折 -> 出 OOF 预测 -> 在 OOF 上选融合参数 -> 用全部成员跑 test -> 出提交包
#
# 用法：
#   .\run_big_ensemble.ps1                       # 假设 5 折都已训完
#   .\run_big_ensemble.ps1 -TrainFolds           # 先把 fold1~4 训到 100 epoch（很耗时）
#   .\run_big_ensemble.ps1 -Scales 1024 1280     # 指定 OOF 推理尺度
#
# 产物：
#   preds/oof/<tag>_s<scale>/fold<k>/*.txt   每折的折外预测
#   preds/oof/tune_results.json              融合参数搜索（按 OOF mAP50 排序）
#   submissions/submit_big_<日期>_*.zip      2 个候选（最佳配置 + 单尺度对照）
param(
    [int[]]$Scales = @(896, 1024),
    [switch]$TrainFolds,
    [int]$FoldEpochs = 100,
    [int]$FoldBatch = 12
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Path (Join-Path $logs ("bigens_" + (Get-Date -Format "MMdd_HHmm") + ".log")) -Append | Out-Null
function Banner($t) { Write-Host "`n===== $t =====" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 0 补齐 5 折
if ($TrainFolds) {
    Banner "0 补齐 fold1~4"
    foreach ($f in 1..4) {
        $done = "runs\detect\fold${f}_y11s_1024\weights\best.pt"
        if (Test-Path $done) { Write-Host "[skip] fold$f 已完成"; continue }
        # 已训到一半的（partial 目录）优先续训
        $partial = Get-ChildItem "runs\detect" -Directory | Where-Object { $_.Name -like "fold${f}_y11s_1024_partial_ep*" } |
            Sort-Object { [int]($_.Name -replace '.*_ep', '') } | Select-Object -Last 1
        if ($partial) {
            Write-Host "[resume] fold$f <- $($partial.Name)"
            & $env:WHEAT_PY train.py --resume-from "$($partial.FullName)\weights\last.pt"
        }
        else {
            & $env:WHEAT_PY train.py --model weights/yolo11s.pt --data "data/folds/fold$f.yaml" `
                --epochs $FoldEpochs --imgsz 1024 --batch $FoldBatch --cache ram `
                --close-mosaic 15 --patience 30 --name "fold${f}_y11s_1024"
        }
    }
}

# ---------------------------------------------------------------- 1 出 OOF 预测
Banner "1 生成各折 OOF 预测（尺度 $($Scales -join ',')）"
$oof = "preds\oof"
New-Item -ItemType Directory -Force -Path $oof | Out-Null
foreach ($f in 0..4) {
    $w = "runs\detect\fold${f}_y11s_1024\weights\best.pt"
    if (-not (Test-Path $w)) { Write-Host "[warn] fold$f 没有权重，跳过"; continue }
    foreach ($s in $Scales) {
        $tag = "fold${f}_y11s_s$s"
        $dst = Join-Path $oof "$tag\fold$f"
        $n = (Get-ChildItem $dst -Filter *.txt -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($n -gt 50) { Write-Host "[skip] $tag 已存在 $n 个 txt"; continue }
        Write-Host "[oof] $tag"
        & $env:WHEAT_PY scripts\predict.py --weights $w --source-list "data/folds/fold${f}_val.txt" `
            --out $dst --imgsz $s --conf 0.001 --iou 0.6 --batch 8
    }
}

# 大模型成员在"它们没见过的图片"上没有 OOF，只能用折模型做融合选参（这是正确做法）
Banner "2 在 OOF 上搜索融合参数"
& $env:WHEAT_PY scripts\tune_fusion_oof.py --root $oof --folds 0 1 2 3 4 `
    --fuse nms wbf --iou 0.5 0.55 0.6 0.65 --box-scale 1.0 1.03 0.97

# ---------------------------------------------------------------- 3 大集成跑 test
Banner "3 全部成员跑 test"
$members = @()
foreach ($n in @("long_y11l_1024", "x900_1024", "l1280_900", "l_v2_1280", "m900_1024", "l_v2_900",
        "fold0_y11s_1024", "fold1_y11s_1024", "fold2_y11s_1024", "fold3_y11s_1024", "fold4_y11s_1024")) {
    $w = "runs\detect\$n\weights\best.pt"
    if (Test-Path $w) {
        $scales = if ($n -like "*1280*") { @(1280, 1024) } else { @(1024, 896) }
        $weight = if ($n -like "fold*") { 0.8 } else { 1.0 }
        $members += [pscustomobject]@{ path = $w; scales = $scales; weight = $weight }
    }
}
if (Test-Path "weights\x_soup.pt") {
    $members += [pscustomobject]@{ path = "weights/x_soup.pt"; scales = @(1024, 896); weight = 1.0 }
}
$members | ConvertTo-Json -Depth 4 | Set-Content -Path "members.json" -Encoding utf8
Write-Host "[big] 成员 $($members.Count) 个 -> members.json"

$stamp = Get-Date -Format "MMdd"
$tune = Get-Content "preds\oof\tune_results.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
if ($tune) {
    $best = $tune[0]
    Write-Host "[big] 采用 OOF 最优: fuse=$($best.fuse) iou=$($best.iou) box_scale=$($best.box_scale) (OOF mAP50=$($best.mAP50))"
    & $env:WHEAT_PY scripts\ensemble_big.py --members members.json --source data/dataset/images/test `
        --out "preds\test_big_$stamp" --fuse $best.fuse --nms-iou $best.iou --wbf-iou $best.iou `
        --box-scale $best.box_scale --tag "big_${stamp}_$($best.fuse)"
}
else {
    Write-Host "[warn] 没有 tune_results.json，用默认参数（WBF 0.55）"
    & $env:WHEAT_PY scripts\ensemble_big.py --members members.json --source data/dataset/images/test `
        --out "preds\test_big_$stamp" --fuse wbf --wbf-iou 0.55 --tag "big_${stamp}_wbf"
}

Banner "完成"
Get-ChildItem submissions\*.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime |
    Select-Object -Last 3 Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 2) } }
Stop-Transcript | Out-Null
