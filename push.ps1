# 一键推送：. \push.ps1            （推当前分支到 origin）
#          . \push.ps1 -Msg "说明"  （先提交全部改动再推送）
param([string]$Msg)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
. "$PSScriptRoot\env.ps1" | Out-Null

if ($Msg) {
    git add -A
    git commit -m $Msg
}

$branch = git rev-parse --abbrev-ref HEAD
Write-Host "[push] branch=$branch remote=$(git remote get-url origin)"

# 首次推送需要凭据：Username 填 GitHub 用户名，Password 填 Personal Access Token
# （GitHub 已不支持账号密码；token 会被 wincred 存进 Windows 凭据管理器）
git push -u origin $branch
exit $LASTEXITCODE
