$ErrorActionPreference = 'Stop'
$taskLab = $PSScriptRoot
$taskPython = Join-Path $taskLab '.venv-yolo\Scripts\python.exe'
$taskBatchId = 'batch_001'
$taskPort = 8765
$taskActiveFile = Join-Path $taskLab 'data\review_batches\active_batch.json'
if (Test-Path -LiteralPath $taskActiveFile) {
    $taskActive = Get-Content -LiteralPath $taskActiveFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskBatchId = [string]$taskActive.batch_id
    $taskPort = [int]$taskActive.port
}
if ($taskBatchId -notmatch '^[A-Za-z0-9_-]+$' -or $taskPort -lt 1024 -or $taskPort -gt 65535) {
    throw '当前批次配置无效，请检查 data/review_batches/active_batch.json。'
}
$taskBatchRelative = "data/review_batches/$taskBatchId"
$taskBatch = Join-Path $taskLab "$taskBatchRelative/batch.json"
$taskUrl = "http://127.0.0.1:$taskPort"
if (-not (Test-Path -LiteralPath $taskPython)) { throw '没有找到本项目的 Python 环境。' }
if (-not (Test-Path -LiteralPath $taskBatch)) { throw "没有找到 $taskBatchId 预标注数据。" }
$taskReady = $false
$taskState = $null
try {
    $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
} catch [System.Net.Http.HttpRequestException] { }
catch [System.Net.WebException] { }
if ($null -ne $taskState) {
    if ($taskState.batch_id -ne $taskBatchId) { throw "$taskPort 端口正在服务其他批次，请检查当前批次配置。" }
    $taskReady = $true
}
if (-not $taskReady) {
    $taskLogDir = Join-Path $taskLab 'outputs\review_workflow'
    New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
    $env:PYTHONUTF8 = '1'
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', 'scripts/review_workflow.py', 'serve', '--batch', $taskBatchRelative, '--port', [string]$taskPort) -WorkingDirectory $taskLab -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogDir "server_${taskBatchId}.log") -RedirectStandardError (Join-Path $taskLogDir "server_${taskBatchId}_error.log") -PassThru
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskLogDir "server_${taskBatchId}.pid")
    for ($taskAttempt = 0; $taskAttempt -lt 20; $taskAttempt++) {
        try {
            $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
            if ($taskState.batch_id -ne $taskBatchId) { throw '服务返回的批次与配置不一致。' }
            $taskReady = $true
            break
        } catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $taskReady) { throw "本地复核页面启动失败，请查看 outputs/review_workflow/server_${taskBatchId}_error.log。" }
}
Start-Process $taskUrl
Write-Host "元件复核页面已打开。数据保存在本项目 $taskBatchRelative。"
