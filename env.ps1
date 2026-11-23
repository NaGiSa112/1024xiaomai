# Wheat-SOD-train  |  environment activation (dot-source this file)
#   . .\env.ps1
#
# Why not a venv/pip: this machine's workspace ACL breaks pip's atomic rename,
# so the torch/ultralytics stack is a pre-extracted package tree under .\pkgs
# (built for CPython 3.12 / cp312 wheels, torch 2.6.0+cu124).
#
# NOTE: paths are derived from $PSScriptRoot on purpose -- the workspace path
# contains non-ASCII characters, and a UTF-8 file without BOM would be read as
# ANSI by Windows PowerShell and corrupt them.

$WHEAT_ROOT = $PSScriptRoot

$DSH_PY = "C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

$env:PYTHONPATH       = Join-Path $WHEAT_ROOT "pkgs"
$env:YOLO_CONFIG_DIR  = Join-Path $WHEAT_ROOT "_cfg\ultralytics"
$env:MPLCONFIGDIR     = Join-Path $WHEAT_ROOT "_cfg\mpl"
$env:OMP_NUM_THREADS  = "4"
$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:WHEAT_PY         = $DSH_PY
$env:WHEAT_ROOT       = $WHEAT_ROOT

# 本机 Python 的默认 CA 缺中间证书, 直连 github/zenodo 会报
#   CERTIFICATE_VERIFY_FAILED: unable to get issuer certificate
# 指向 certifi 的证书包后 api.github.com / huggingface.co 均正常
$CERTIFI = Join-Path $WHEAT_ROOT "pkgs\certifi\cacert.pem"
if (Test-Path $CERTIFI) { $env:SSL_CERT_FILE = $CERTIFI }

# 便携版 git（MinGit, 无需管理员安装）
$GIT_CMD = Join-Path $WHEAT_ROOT "tools\git\cmd"
if (Test-Path (Join-Path $GIT_CMD "git.exe")) {
    $env:PATH = "$GIT_CMD;$env:PATH"
}

Write-Host "[env] root   : $env:WHEAT_ROOT"
Write-Host "[env] python : $env:WHEAT_PY"
Write-Host "[env] pkgs   : $env:PYTHONPATH"
if ($env:SSL_CERT_FILE) { Write-Host "[env] ssl    : $env:SSL_CERT_FILE" }
Write-Host "[env] git    : $(if (Get-Command git -ErrorAction SilentlyContinue) { (Get-Command git).Source } else { 'MISSING' })"
