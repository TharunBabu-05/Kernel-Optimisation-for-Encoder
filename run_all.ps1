# PS-09: one-click setup and run on Windows (PowerShell).
# Usage, from this folder:   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Re-running is safe: finished steps are skipped.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force results | Out-Null

Write-Host "== 1. GPU" -ForegroundColor Cyan
nvidia-smi | Tee-Object results\nvidia_smi.txt

Write-Host "== 2. Python environment" -ForegroundColor Cyan
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install transformers numpy huggingface_hub
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

Write-Host "== 3. Dataset" -ForegroundColor Cyan
if (-not (Test-Path data\practice_passages.jsonl)) {
  & $py -c "from huggingface_hub import snapshot_download; snapshot_download('Tarakeshwaran/CIT_Hackathon_PS09_encoder', repo_type='dataset', local_dir='data')"
}

Write-Host "== 4. Reference baseline (first 5,000 passages)" -ForegroundColor Cyan
if (-not (Test-Path results\ref_throughput.json)) {
  $out = & $py data\reference.py --passages data\practice_passages.jsonl --out results\ref5k.npy --limit 5000 2>$null
  ($out | Select-Object -Last 1) | Set-Content results\ref_throughput.json
}
Get-Content results\ref_throughput.json

Write-Host "== 5. Accuracy gate (stage 4)" -ForegroundColor Cyan
$env:PS09_STAGE = "4"; $env:PS09_DEBUG = "1"
& $py validate.py --data data --out results\validate_stage4.json
Remove-Item Env:PS09_DEBUG

Write-Host "== 6. Ablation with the official bench.py (stages 0-4; about 4 min of live trace each)" -ForegroundColor Cyan
& $py run_ablation.py --data data --stages 0 1 2 3 4

Write-Host "== 7. Profiler evidence" -ForegroundColor Cyan
& $py profile_run.py --which reference --data data --n 2000 --out results\prof_reference
$env:PS09_STAGE = "4"
& $py profile_run.py --which team --data data --n 2000 --out results\prof_stage4

Write-Host "== Done. Send results\ABLATION.md, results\validate_stage4.json and results\nvidia_smi.txt" -ForegroundColor Green
