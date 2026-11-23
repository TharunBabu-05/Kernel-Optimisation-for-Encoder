# PS-09 implementation tracker

Status key: ✅ done · 🔄 in progress · ⏳ waiting on GPU · ⬜ not started

## Phases

| # | Phase | Optimisation | Status | Deliverable | Result |
|---|---|---|---|---|---|
| 0 | Analysis and planning | — | ✅ | Plan v2, explainer, problem list | Mean 100.2 tokens; 80.4% padding; 5.2× FLOP gain available; tokenising 4.1 s/100k |
| 1 | Code and correctness harness | — | ✅ | `team.py`, `validate.py`, `profile_run.py`, `run_ablation.py` | CPU FP32 check: stages 0, 2 and 4 match the reference (min cosine 1.000000) |
| 2 | GPU baseline and "before" profile | Step 0 | 🔄 baseline done | `results/ref_throughput.json`, `results/prof_reference/` | — |
| 3 | S1: FP16, fused QKV, SDPA | O1 | ⏳ code ready | `bench_stage1.json`, validate | — |
| 4 | S2: length-aware token-budget scheduler | O2 (Innovation A) | ⏳ code ready | `bench_stage2.json` | CPU dry run: 4.55× (FP32, no GPU) |
| 5 | S3: async host pipeline and CUDA Graph live path | O3, O7 (Innovation C) | ⏳ code ready | `bench_stage3.json`, live p50/p99 | — |
| 6 | S4: CLS-only last layer | O5 | ⏳ code ready | `bench_stage4.json` | CPU dry run: 4.83× (+6% over S2, as calculated) |
| 7 | S5: torch.compile | O4 | ⏳ code ready | `bench_stage5.json` | Keep only if faster |
| 8 | Budget sweep, full 100k validation, "after" profiles | — | ⬜ | `ablation.csv`, `prof_stage4/` | — |
| 9 | Report and demo | — | ⬜ | Profiler report PDF | — |

## Results log

| When (IST) | Where | What | Result |
|---|---|---|---|
| 23 Nov, 15:40 | Cloud CPU, FP32 | `validate.py` stage 4, 300 passages + 60 queries | min cosine 1.000000 / 1.000000 → PASS |
| 23 Nov, 15:45 | Cloud CPU, FP32 | `validate.py` stages 0 and 2, 100 passages + 20 queries | min cosine 1.000000 → PASS |

| 23 Nov, 15:55 | Cloud CPU, FP32 | Official `bench.py` via `run_ablation.py`, first 300 passages, budget 8192 | S0 4.0 p/s (1.02×) · S2 17.8 p/s (**4.55×**) · S4 18.9 p/s (**4.83×**); 0 violations, min cos 1.0. Files in `results_cpu_dryrun/` |
| 23 Nov, 16:55 | User's Linux RTX machine | `run_all.sh` steps 1–4 (GPU check, install, dataset, baseline) | OK. Step 5 crashed: debug logging read the GPU during CUDA Graph capture. Fixed in `team.py` line 115 |
| 23 Nov, 17:00 | Cloud CPU, FP32 | Re-test after fix, `PS09_DEBUG=1` | PASS, min cos 1.000000; max abs activation 16.7 (FP16 limit 65504, no overflow risk) |
| 23 Nov, 17:42 | RTX 3050 6GB Laptop, Linux | `nvidia-smi`, reference baseline (5,000 passages, FP32, pad 512) | **28.4 passages/s** (176 s) |
| 23 Nov, 18:05 | RTX 3050 6GB Laptop, Linux | `validate.py` stage 4 (FP16), 2,000 passages + 1,500 queries | passages min cos **0.999663** (p0.1 0.999987); queries min **0.999982**; max activation 16.4 → **PASS** |
| 23 Nov, 18:10 | RTX 3050 6GB Laptop, Linux | Ablation stage 0 on 100k | Stopped by user: FP32 pad-512 over 100k takes about 1 h at 28 p/s. Stage 0 removed from the run; the reference speed comes from the baseline above |

CPU dry-run note: this proves the scheduler and CLS-only gains are real and exact, but CPU numbers are not the score. FP16 Tensor Cores (S1), the async pipeline and CUDA Graphs only show their effect on a GPU.

## Decisions

- Dev GPU: NVIDIA GeForce RTX, now on a Linux machine (exact model and VRAM to confirm with nvidia-smi). On Linux all stages 0–5 can run, including torch.compile.

- O6 (padding-free varlen attention) dropped: residual padding after sorting is only 4%.
- Live path: `bench.py` calls `encode([q])` one at a time, so no cross-query batching. A CUDA Graph per bucket (32/64/128) is used instead.
- The internal accuracy gate is a minimum cosine ≥ 0.9995; the official limit is 0.999.
