#!/usr/bin/env bash
# PS-09: one-shot setup and run from a bash shell (Git Bash on Windows, WSL, or Linux).
# Usage, from this folder:   bash run_all.sh            (re-running is safe: finished steps are skipped)
set -e
cd "$(dirname "$0")"
mkdir -p results

echo "== 1. GPU"
nvidia-smi | tee results/nvidia_smi.txt

echo "== 2. Python environment"
PYBIN=$(command -v python3 || command -v python)
[ -d .venv ] || "$PYBIN" -m venv .venv
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
"$PY" -m pip install --upgrade pip
if [ -f .venv/Scripts/python.exe ]; then
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
else
  "$PY" -m pip install torch
fi
"$PY" -m pip install transformers numpy huggingface_hub
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

echo "== 3. Dataset"
[ -f data/practice_passages.jsonl ] || "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Tarakeshwaran/CIT_Hackathon_PS09_encoder', repo_type='dataset', local_dir='data')"

echo "== 4. Reference baseline (first 5,000 passages)"
[ -s results/ref_throughput.json ] || "$PY" data/reference.py --passages data/practice_passages.jsonl --out results/ref5k.npy --limit 5000 2>/dev/null | tail -1 > results/ref_throughput.json
cat results/ref_throughput.json

echo "== 5. Accuracy gate (stage 4)"
PS09_STAGE=4 PS09_DEBUG=1 "$PY" validate.py --data data --out results/validate_stage4.json

echo "== 6. Ablation with the official bench.py"
# Stage 0 (FP32, pad 512) over 100k passages takes about an hour on a laptop GPU; the reference speed is already
# measured in step 4, so it is skipped here. Quick ablation on 20k passages first, then the full 100k run.
"$PY" run_ablation.py --data data --stages 1 2 --limit 20000 --no-trace
"$PY" run_ablation.py --data data --stages 3 4 --limit 20000
"$PY" run_ablation.py --data data --stages 4
case "$(uname -s)" in Linux*) "$PY" run_ablation.py --data data --stages 5 --limit 20000 --no-trace;; esac

echo "== 7. Profiler evidence"
"$PY" profile_run.py --which reference --data data --n 2000 --out results/prof_reference
PS09_STAGE=4 "$PY" profile_run.py --which team --data data --n 2000 --out results/prof_stage4

echo "== Done. Results are in results/ (ABLATION.md, validate_stage4.json, nvidia_smi.txt, prof_*/)"
