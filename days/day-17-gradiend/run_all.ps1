# Run Day 17 runs one after another with 4 CPU threads (a GRADIEND over BERT needs ~8 GB RAM). Usage: run_all.ps1 -Runs a,b [-Cache]
param([string]$Runs, [switch]$Cache)
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($Cache) { $env:GRADIEND_USE_CACHE = "1" }
$py = "$env:USERPROFILE\.venvs\gradiend\Scripts\python.exe"
Set-Location $PSScriptRoot
foreach ($r in ($Runs -split ',')) { & $py gradiend_probe.py $r *> "run_$r.log" }
