# Hertzon setup: creates .venv next to this script and installs Hertzon with all dependencies.
# Started automatically by Hertzon.exe on first run; can also be run by hand.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Hertzon setup'

function Find-Python {
    try { $p = & py -3.13 -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    # a Python 3.13 on PATH (for example one installed without the py launcher)
    try { $p = & python -c "import sys; print(sys.executable if sys.version_info[:2] == (3, 13) else '')" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
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
    # EMerge 3 shows a one-time welcome text at the first import in a console and waits for Enter. The input
    # comes from a pipe here (no prompt, the setup runs unattended in the release check), and the welcome is
    # marked as shown, so Hertzon.exe and the warm-up console never stop at it either.
    '' | & $venvPy -c "import emerge; from emerge._emerge.global_cache import EMergeGlobalCache; EMergeGlobalCache().set_user_warned('welcome_message'); print('EMerge', getattr(emerge, '__version__', 'ok'))"
    if ($LASTEXITCODE -ne 0) { throw 'EMerge could not be imported.' }
    # compile all solver kernels now with a small test solve, so the first real solve starts at full speed
    '' | & $venvPy -m hertzon.solver.warmup
    if ($LASTEXITCODE -ne 0) { Write-Host 'Solver kernels could not be prepared now; they are compiled on the first solve.' -ForegroundColor Yellow }
    Set-Content -LiteralPath (Join-Path $PSScriptRoot '.venv\hertzon_installed.txt') -Value (Get-Date)
    Write-Host ''
    Write-Host 'Hertzon is installed. Start it with Hertzon.exe.' -ForegroundColor Green
    Start-Sleep -Seconds 3
    exit 0
} catch {
    Write-Host ''
    Write-Host "ERROR: $_" -ForegroundColor Red
    if (-not $env:CI) { Read-Host 'Press Enter to close' }   # the release check runs it unattended
    exit 1
}
