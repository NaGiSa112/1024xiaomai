# 续跑方案（08:10 -> 13:00）：补 fold2/3/4 -> 训 l_v2@1280 -> OOF 调参 + 大集成
#
# 前提：run_overnight.ps1 的最后一相（fold1 + 集成）会在 08:10 左右结束，本脚本先等它跑完。
#
# 用法：
#   .\run_extend.ps1 -Deadline "12:45"
#   .\run_extend.ps1 -Deadline "12:45" -FoldEpochs 60 -DoL1280:$false
param(
    [string]$Deadline = "12:45",
    [int]$FoldEpochs = 70,
    [int]$FoldBatch = 12,
    [switch]$SkipL1280,
    [int]$L1280Epochs = 35
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Path (Join-Path $logs ("extend_" + (Get-Date -Format "MMdd_HHmm") + ".log")) -Append | Out-Null

function Banner($t) { Write-Host "`n===== $t =====" -ForegroundColor Cyan }
$dl = [datetime]::Today + [timespan]::Parse($Deadline)
if ($dl -le (Get-Date)) { $dl = $dl.AddDays(1) }
function Remaining { [math]::Round(($dl - (Get-Date)).TotalHours, 2) }

# ---------------------------------------------------------------- 等上一轮跑完
Banner "等待 run_overnight 收尾（fold1 + 集成）"
$waited = 0
while ($waited -lt 3600) {
    $py = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
    $overnight = Get-Process pwsh -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -lt (Get-Date).AddHours(-1) } | Measure-Object
    if ($py -eq 0) { Write-Host "[wait] GPU 已空闲（等了 $([math]::Round($waited/60,1)) 分钟）"; break }
    Start-Sleep -Seconds 30
    $waited += 30
}
Write-Host "[extend] start $((Get-Date).ToString('HH:mm'))  deadline $dl  剩余 $(Remaining) h"

# ---------------------------------------------------------------- 1 补 fold2/3/4
foreach ($f in 2..4) {
    if ((Remaining) -lt 1.2) { Write-Host "[skip] 剩余不足，停止补折"; break }
    $done = "runs\detect\fold${f}_y11s_1024\weights\best.pt"
    if (Test-Path $done) { Write-Host "[skip] fold$f 已完成"; continue }
    $last = "runs\detect\fold${f}_y11s_1024\weights\last.pt"
    Banner "补 fold$f -> $FoldEpochs ep"
    if (Test-Path $last) {
        & $env:WHEAT_PY train.py --resume-from $last
    }
    else {
        & $env:WHEAT_PY train.py --model weights/yolo11s.pt --data "data/folds/fold$f.yaml" `
            --epochs $FoldEpochs --imgsz 1024 --batch $FoldBatch --cache ram `
            --close-mosaic 12 --patience 25 --name "fold${f}_y11s_1024"
    }
}

# ---------------------------------------------------------------- 2 l 用 v2 预训练权重再训一个 @1280
if (-not $SkipL1280 -and (Remaining) -gt 1.3 -and -not (Test-Path "runs\detect\l_v2_1280\weights\best.pt")) {
    Banner "l_v2 @1280（从 l_pre_v2_768 热启动）x$L1280Epochs"
    $init = if (Test-Path "runs\detect\l_pre_v2_768\weights\best.pt") { "runs/detect/l_pre_v2_768/weights/best.pt" } else { "weights/yolo11l.pt" }
    & $env:WHEAT_PY train.py --model $init --data data/wheat_full900.yaml `
        --epochs $L1280Epochs --imgsz 1280 --batch 6 --cache ram `
        --close-mosaic 8 --patience 20 --seed 21 --name "l_v2_1280"
}
else { Write-Host "[skip] l_v2@1280" }

# ---------------------------------------------------------------- 3 OOF 调参 + 大集成
Banner "OOF 调参 + 大集成"
& "$PSScriptRoot\run_big_ensemble.ps1" -Scales 896 1024

Banner "结束"
Get-ChildItem submissions\*.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime |
    Select-Object -Last 3 Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 2) } }
Stop-Transcript | Out-Null
