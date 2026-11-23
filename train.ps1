# 用法: .\train.ps1 -Args "train.py","--model","weights/yolo11s.pt","--epochs","100"
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

& $env:WHEAT_PY @Args
exit $LASTEXITCODE
