# PS-09 optimised BGE encoder

Submission file: **`team.py`** (defines `setup()` and `encode(texts)`).

| File | Purpose |
|---|---|
| `team.py` | The optimised encoder. Stage presets `PS09_STAGE=0..5` switch each optimisation on, for the ablation. Default is stage 4. |
| `validate.py` | Accuracy gate (Innovation B): worst-case cosine against the reference passages and queries. |
| `profile_run.py` | Profiler evidence: Chrome trace, top kernels, GPU-time split by category. |
| `run_ablation.py` | Runs the official `bench.py` for each stage, and appends to `results/ablation.csv` and `results/ABLATION.md`. |
| `TRACKER.md` | Implementation phases and results log. |

## Run on the GPU machine

```bash
pip install torch transformers numpy          # a CUDA build of torch
# DATA = folder with the dataset files (bench.py, reference.py, practice_*.jsonl, practice_ref_*.npy)
huggingface-cli download Tarakeshwaran/CIT_Hackathon_PS09_encoder --repo-type dataset --local-dir DATA

# 1. Baseline throughput of the reference on this GPU (first 5,000 passages is enough)
python DATA/reference.py --passages DATA/practice_passages.jsonl --out results/ref5k.npy --limit 5000 | tail -1 > results/ref_throughput.json

# 2. Accuracy gate for the default stage (add PS09_DEBUG=1 to log the largest activation)
python validate.py --data DATA --out results/validate_stage4.json

# 3. Ablation with the official harness (the live trace replays in real time and takes about 4 minutes per stage)
python run_ablation.py --data DATA --stages 0 1 2 3 4 5

# 4. Profiler evidence, before and after
python profile_run.py --which reference --data DATA --n 2000 --out results/prof_reference
PS09_STAGE=4 python profile_run.py --which team --data DATA --n 2000 --out results/prof_stage4

# Token-budget sweep (pick the fastest one that passes)
for B in 16384 32768 65536 131072; do PS09_TOKEN_BUDGET=$B python run_ablation.py --data DATA --stages 4 --no-trace; done
```

Switches (environment variables): `PS09_STAGE`, `PS09_FP16`, `PS09_SORT`, `PS09_BUDGET`, `PS09_OVERLAP`,
`PS09_CLS_ONLY`, `PS09_COMPILE`, `PS09_GRAPH`, `PS09_TOKEN_BUDGET`, `PS09_CHUNK`, `PS09_MODEL` (a local model folder
works too), `PS09_DEBUG`.

Before submitting, copy the winning settings into the defaults at the top of `team.py`, because the organizers will not set environment variables.
