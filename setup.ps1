# Hertzon setup: creates .venv next to this script and installs Hertzon with all dependencies.
# Started automatically by Hertzon.exe on first run; can also be run by hand.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Hertzon setup'

function Find-Python {
    try { $p = & py -3.13 -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe", "$env:ProgramFiles\Python313\python.exe")) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return $null
}

try {
    $venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPy)) {
        $py = Find-Python
        if (-not $py) {
            Write-Host 'Python 3.13 not found - installing it with winget ...'
            winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
            $py = Find-Python
            if (-not $py) { throw 'Python 3.13 could not be installed automatically. Install it from python.org and start again.' }
        }
        Write-Host "Using $py"
        & $py -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
    }
    Write-Host 'Installing Hertzon and its packages (EMerge, Qt, VTK) ...'
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -e ".[test]"
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed (see messages above).' }
    Write-Host 'Preparing the solver (first-time compilation, can take a few minutes) ...'
    & $venvPy -c "import emerge; print('EMerge', emerge.__version__ if hasattr(emerge, '__version__') else 'ok')"
    if ($LASTEXITCODE -ne 0) { throw 'EMerge could not be imported.' }
    # compile all solver kernels now with a small test solve, so the first real solve starts at full speed
    & $venvPy -m hertzon.solver.warmup
    if ($LASTEXITCODE -ne 0) { Write-Host 'Solver kernels could not be prepared now; they are compiled on the first solve.' -ForegroundColor Yellow }
    Set-Content -LiteralPath (Join-Path $PSScriptRoot '.venv\hertzon_installed.txt') -Value (Get-Date)
    Write-Host ''
    Write-Host 'Hertzon is installed. Start it with Hertzon.exe.' -ForegroundColor Green
    Start-Sleep -Seconds 3
    exit 0
} catch {
    Write-Host ''
    Write-Host "ERROR: $_" -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
