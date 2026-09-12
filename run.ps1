<#  Launch SingCoach.  #>
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $py)) { $py = Join-Path $root '.venv\Scripts\python.exe' }
if (-not (Test-Path $py)) { throw "Environment not set up. Run .\scripts\setup.ps1 first." }
$env:PYTHONPATH = $root
& $py -m singcoach @args
