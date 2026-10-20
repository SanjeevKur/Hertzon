param([string]$Project = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host 'First start: running setup.ps1 ...'
        & (Join-Path $PSScriptRoot 'setup.ps1')
        if (-not (Test-Path -LiteralPath $python)) { throw 'Setup did not create .venv - see messages above.' }
    }
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Project) { & $python -m hertzon $Project } else { & $python -m hertzon }
    if ($LASTEXITCODE -ne 0) { throw "Hertzon exited with code $LASTEXITCODE" }
} catch {
    Write-Host ''
    Write-Host "ERROR: $_" -ForegroundColor Red
    Read-Host 'Press Enter to close'
}
