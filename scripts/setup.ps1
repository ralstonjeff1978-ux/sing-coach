<#
    SingCoach environment setup.

    Run this instead of `pip install -r requirements.txt`. The extra step at the
    end is not optional — see the comment block at the top of requirements.txt
    for why the DirectML runtime has to be installed last.
#>

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPy)) {
    if (-not (Test-Path $py312)) { throw "Python 3.12 not found at $py312" }
    Write-Host 'Creating virtualenv (Python 3.12)...' -ForegroundColor Cyan
    & $py312 -m venv (Join-Path $root '.venv')
}

Write-Host 'Installing dependencies...' -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r (Join-Path $root 'requirements.txt')

# faster-whisper drags in stock onnxruntime, which shadows the DirectML build.
# Reinstall DirectML last, then repair numpy (force-reinstall bumps it past
# numba's ceiling).
Write-Host 'Restoring DirectML ONNX runtime...' -ForegroundColor Cyan
& $venvPy -m pip uninstall -y onnxruntime onnxruntime-directml
& $venvPy -m pip install --no-cache-dir onnxruntime-directml
& $venvPy -m pip install "numpy<2.5"

Write-Host 'Verifying...' -ForegroundColor Cyan
& $venvPy -c @"
import onnxruntime as ort, sys
providers = ort.get_available_providers()
print('ONNX providers:', providers)
if 'DmlExecutionProvider' not in providers:
    print('WARNING: DirectML unavailable - separation will run on CPU (much slower).')
    sys.exit(1)
import numpy, numba, librosa, soundfile, sounddevice, PySide6, pyqtgraph, pedalboard, faster_whisper
print('All imports OK. GPU separation enabled.')
"@

Write-Host 'Setup complete.' -ForegroundColor Green
