# 早上更新仓库：刷新实验记录 -> 汇总本次训练 -> 提交
#
# 用法：
#   .\update_repo.ps1                  # 刷新实验记录 + 提交
#   .\update_repo.ps1 -Tag "1125"      # 生成 docs\跨夜结果_1125.md 并提交
#   .\update_repo.ps1 -Push            # 顺便 git push
param(
    [string]$Tag = (Get-Date -Format "MMdd"),
    [switch]$Push
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

Write-Host "=== 1) 刷新实验记录 ==="
& $env:WHEAT_PY scripts\make_experiment_log.py

Write-Host "=== 2) 汇总本次训练 ==="
$rows = @()
foreach ($d in Get-ChildItem "runs\detect" -Directory | Sort-Object LastWriteTime) {
    $s = Join-Path $d.FullName "summary.json"
    if (Test-Path $s) {
        try { $j = Get-Content $s -Raw | ConvertFrom-Json } catch { continue }
        $rows += [pscustomobject]@{
            run     = $d.Name
            epochs  = $j.epochs
            mAP50   = $j.'metrics/mAP50(B)'
            mAP5095 = $j.'metrics/mAP50-95(B)'
            minutes = $j.minutes
        }
    }
}
$rows | Sort-Object mAP50 -Descending | Format-Table -AutoSize | Out-String -Width 100 | Write-Host

$md = @("# 跨夜训练结果（$Tag）", "", "| run | epoch | mAP50 | mAP50-95 | 耗时(h) |", "|---|---|---|---|---|")
foreach ($r in ($rows | Sort-Object mAP50 -Descending)) {
    $md += ("| ``{0}`` | {1} | {2:N4} | {3:N4} | {4:N2} |" -f $r.run, $r.epochs, $r.mAP50, $r.mAP5095, ($r.minutes / 60))
}
$subs = Get-ChildItem "submissions\*.zip" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 3
if ($subs) {
    $md += @("", "## 提交候选", "", "| 文件 | 大小(MB) |", "|---|---|")
    foreach ($s in $subs) { $md += ("| ``{0}`` | {1:N2} |" -f $s.Name, ($s.Length / 1MB)) }
}
$md += @("", "> 指标说明：用 ``wheat_full900.yaml`` 训练的 run，其 val 含官方 val（污染），只能看收敛；干净口径看 ``fold*`` 与 ``*_pre_*``。")
$outMd = "docs\跨夜结果_$Tag.md"
$md -join "`n" | Set-Content -Path $outMd -Encoding utf8
Write-Host "写出 $outMd"

Write-Host "=== 3) 提交 ==="
git add -A
git -c core.autocrlf=false commit -m "chore: 跨夜训练结果 $Tag（实验记录 + 结果汇总）" | Out-Null
git log --oneline -1
Write-Host ""
Write-Host "下一步："
Write-Host "  1) 到比赛页面提交 submissions 下最新的 zip，把分数告诉我（我写进 docs/提交记录.md）"
Write-Host "  2) 推送：git push -u origin main   （或 .\push.ps1）"

if ($Push) {
    $branch = git rev-parse --abbrev-ref HEAD
    git push -u origin $branch
}
