# 长训练 v2：GWHD-clean 预训练 -> 赛题 900 微调 -> 收尾集成
#
# 背景：赛题数据 47%~58% 与 GWHD 同源（同一张图的重编码），已剔除 539 张同源图，
#       剩下 2834 张是同分布外部数据 => 预训练是最划算的杠杆。
#
# 用法：
#   .\run_long.ps1 -Deadline "08:00"                          # 00:10 开跑的标准流程
#   .\run_long.ps1 -Deadline "08:00" -SkipPre                 # 只用 900 张（对照组）
#   .\run_long.ps1 -Deadline "08:00" -Second -SecondEpochs 70  # 追加第二个模型
#
# 阶段：
#   0  体检：干净清单存在性 + 数量
#   1  预训练：yolo11l @768 on 2834 张 GWHD-clean（40 ep）
#   2  微调：yolo11l @1024 on 900 张赛题数据（140 ep，含 val）
#   3  第二模型：yolo11m @1024 on 900（可选，剩余 > 2.5h 才做）
#   4  收尾：全部模型的 test 多尺度推理 -> NMS 融合 -> submit.zip（不自动提交）
#
# 断点：任一步中断，重跑本脚本即从 last.pt 续训，不会从头再来。
param(
    [string]$Deadline = "08:00",
    [int]$PreEpochs = 40,
    [int]$PreImgsz = 768,
    [int]$PreBatch = 12,
    [int]$FineEpochs = 140,
    [int]$FineImgsz = 1024,
    [int]$FineBatch = 8,
    [switch]$SkipPre,
    [switch]$Second,
    [int]$SecondEpochs = 70,
    [int]$SecondBatch = 10
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Path (Join-Path $logs ("long_" + (Get-Date -Format "MMdd_HHmm") + ".log")) -Append | Out-Null

$dl = [datetime]::Today + [timespan]::Parse($Deadline)
if ($dl -le (Get-Date)) { $dl = $dl.AddDays(1) }
function Remaining { [math]::Round(($dl - (Get-Date)).TotalHours, 2) }
function Banner($t) { Write-Host "`n===== $t (剩余 $(Remaining) h) =====" -ForegroundColor Cyan }

function Invoke-Train($name, $model, $data, $epochs, $imgsz, $batch, $extra) {
    $best = "runs\detect\$name\weights\best.pt"
    $last = "runs\detect\$name\weights\last.pt"
    if (Test-Path $best) { Write-Host "[skip] $name 已完成"; return }
    if (Test-Path $last) {
        Write-Host "[resume] $name <- last.pt"
        & $env:WHEAT_PY train.py --resume-from $last
        return
    }
    $argv = @("train.py", "--model", $model, "--data", $data, "--epochs", "$epochs",
              "--imgsz", "$imgsz", "--batch", "$batch", "--cache", "ram", "--name", $name) + $extra
    Write-Host "[train] $name : $($argv[1..($argv.Count - 1)] -join ' ')"
    & $env:WHEAT_PY @argv
}

Write-Host "[long] start $(Get-Date -Format 'HH:mm:ss')  deadline $dl  remaining $(Remaining) h"

# ---------------------------------------------------------------- 显存自适应
# 实测基准：yolo11s@1024 bs12 = 11.1 GB（含 ~1.5 GB CUDA context）
#   => s@1024 单图约 0.80 GB；激活显存近似 ∝ 参数量 × 分辨率² × batch
#   yolo11l ≈ 2.7× s 的 FLOPs：l@768 单图 ≈ 1.21 GB，l@1024 单图 ≈ 2.16 GB
# 预留 3 GB 安全余量（碎片 + 峰值抖动），据此反推 batch 上限
$freeMB = 0
try { $freeMB = [int](((nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1).Trim()) } catch { }
$usableGB = [math]::Max(($freeMB / 1024.0) - 3.0, 2.0)
$capPre = [int][math]::Floor($usableGB / 1.21)
$capFine = [int][math]::Floor($usableGB / 2.16)
$capPre = [math]::Min([math]::Max($capPre, 2), 16)
$capFine = [math]::Min([math]::Max($capFine, 1), 10)
Write-Host ("[vram] 启动时空闲 {0} MB -> 可用 {1:N1} GB -> 预训练上限 bs {2} / 微调上限 bs {3}" -f $freeMB, $usableGB, $capPre, $capFine)
if ($PreBatch -gt $capPre) { Write-Host "[vram] 预训练 batch $PreBatch -> $capPre"; $PreBatch = $capPre }
if ($FineBatch -gt $capFine) { Write-Host "[vram] 微调 batch $FineBatch -> $capFine"; $FineBatch = $capFine }
Write-Host "[vram] 本次使用: 预训练 bs=$PreBatch  微调 bs=$FineBatch"

# ---------------------------------------------------------------- 0 体检
Banner "0 体检"
$cleanYaml = "data_external/gwhd_clean.yaml"
$cleanList = "data_external/gwhd_clean_train.txt"
if (-not $SkipPre) {
    if (-not (Test-Path $cleanList)) { throw "缺少干净外部数据清单，先运行 scripts\build_gwhd_clean.py" }
    $n = (Get-Content $cleanList | Where-Object { $_.Trim() }).Count
    Write-Host "[check] 干净外部图片 $n 张"
    if ($n -lt 2000) { throw "干净外部数据只有 $n 张，异常，停止" }
}
Write-Host "[check] 赛题数据 $((Get-ChildItem data\dataset\images\train\*.png).Count) train + $((Get-ChildItem data\dataset\images\val\*.png).Count) val"

# ---------------------------------------------------------------- 1 预训练
if (-not $SkipPre) {
    Banner "1 预训练 yolo11l @$PreImgsz on GWHD-clean ($PreEpochs ep)"
    Invoke-Train "long_pre_y11l_768" "weights/yolo11l.pt" $cleanYaml $PreEpochs $PreImgsz $PreBatch @("--close-mosaic", "10", "--patience", "40", "--seed", "0")
}
else {
    Write-Host "[skip] -SkipPre：不跑预训练"
}

# ---------------------------------------------------------------- 2 微调
Banner "2 微调 yolo11l @$FineImgsz on 900 ($FineEpochs ep)"
if ($SkipPre) {
    Invoke-Train "long_y11l_1024" "weights/yolo11l.pt" "data/wheat_full900.yaml" $FineEpochs $FineImgsz $FineBatch @("--close-mosaic", "20", "--patience", "40", "--seed", "1")
}
else {
    Invoke-Train "long_y11l_1024" "runs/detect/long_pre_y11l_768/weights/best.pt" "data/wheat_full900.yaml" $FineEpochs $FineImgsz $FineBatch @("--close-mosaic", "20", "--patience", "40", "--seed", "1")
}

# ---------------------------------------------------------------- 3 第二模型
if ($Second -and (Remaining) -gt 2.5) {
    Banner "3 第二模型 yolo11m @1024 on 900 ($SecondEpochs ep)"
    Invoke-Train "long_y11m_1024" "weights/yolo11m.pt" "data/wheat_full900.yaml" $SecondEpochs 1024 $SecondBatch @("--close-mosaic", "15", "--patience", "30", "--seed", "7")
}
else {
    Write-Host "[skip] 第二模型（未开 -Second 或剩余 $(Remaining) h < 2.5h）"
}

# ---------------------------------------------------------------- 4 收尾
Banner "4 收尾：test 推理 + 集成 + 打包"
$weights = @()
foreach ($n in @("long_y11l_1024", "long_y11m_1024", "fold0_y11s_1024")) {
    if ($n -and (Test-Path "runs\detect\$n\weights\best.pt")) { $weights += "runs\detect\$n\weights\best.pt" }
}
foreach ($f in 1..4) {
    if (Test-Path "runs\detect\fold${f}_y11s_1024\weights\best.pt") { $weights += "runs\detect\fold${f}_y11s_1024\weights\best.pt" }
}
Write-Host "[ens] 可用模型 $($weights.Count) 个"

if ($weights.Count -gt 0) {
    $tag = "long_" + (Get-Date -Format "MMdd")
    & $env:WHEAT_PY scripts\predict_ensemble.py --weights @weights `
        --source data/dataset/images/test --out "preds/test_$tag" `
        --imgsz 896 1024 1280 --fuse nms --nms-iou 0.6
    & $env:WHEAT_PY scripts\make_submission.py --pred "preds/test_$tag" --tag $tag
    Write-Host "[done] submissions\submit_$tag.zip（每天 3 次提交，先看本地指标再花）"
}
else {
    Write-Host "[warn] 没有任何可用权重，跳过收尾"
}

Banner "全部结束"
Stop-Transcript | Out-Null
