"""Runs the official bench.py once per stage and appends the results to results/ablation.csv + results/ABLATION.md.

Usage:
  # 1) reference throughput on this GPU (once; --limit keeps it short, throughput is steady after a few batches)
  python DATA/reference.py --passages DATA/practice_passages.jsonl --out results/ref_out.npy --limit 5000 > results/ref_throughput.json
  # 2) all stages
  python run_ablation.py --data DATA --stages 0 1 2 3 4 5 [--limit N]
Each stage runs in a fresh process with PS09_STAGE set. Extra PS09_* variables in the environment are passed through
(e.g. PS09_TOKEN_BUDGET=32768 to sweep the budget).
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {0: "S0 reference-like (FP32, pad 512, batch 32)", 1: "S1 + FP16, fused QKV, SDPA",
         2: "S2 + length-aware token-budget scheduler", 3: "S3 + async host pipeline, CUDA Graph live path",
         4: "S4 + CLS-only last layer", 5: "S5 + torch.compile"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--limit", type=int); ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--ref-throughput", default=str(HERE / "results" / "ref_throughput.json"))
    a = ap.parse_args()
    d = Path(a.data); res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
    rows = []
    for st in a.stages:
        out = res_dir / f"bench_stage{st}.json"
        cmd = [sys.executable, str(d / "bench.py"), "--team", str(HERE / "team.py"),
               "--passages", str(d / "practice_passages.jsonl"), "--ref", str(d / "practice_ref_first2000.npy"),
               "--out", str(out)]
        if not a.no_trace:
            cmd += ["--trace", str(d / "practice_trace.jsonl"), "--ref-queries", str(d / "practice_ref_queries.npy")]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if os.path.exists(a.ref_throughput):
            cmd += ["--ref-throughput", a.ref_throughput]
        env = dict(os.environ, PS09_STAGE=str(st))
        print(f"\n=== {NAMES.get(st, st)} ===\n", " ".join(cmd), flush=True)
        t = time.time()
        p = subprocess.run(cmd, env=env)
        if p.returncode != 0 or not out.exists():
            rows.append({"stage": st, "name": NAMES.get(st), "status": f"FAILED (exit {p.returncode})"}); continue
        r = json.load(open(out))
        live = r.get("live", {})
        rows.append({"stage": st, "name": NAMES.get(st), "status": "ok",
                     "passages_per_s": round(r["passages_per_second"], 1), "seconds": round(r["seconds"], 2),
                     "speedup": round(r["speedup"], 2) if "speedup" in r else "",
                     "min_cos": round(r["min_cosine"], 6) if r.get("min_cosine") is not None else "",
                     "violations": r["n_violations"], "live_p50_ms": round(live.get("p50_ms", float("nan")), 2) if live else "",
                     "live_p99_ms": round(live.get("p99_ms", float("nan")), 2) if live else "",
                     "live_min_cos": round(live["min_cosine"], 6) if live.get("min_cosine") is not None else "",
                     "setup_s": round(r["setup_seconds"], 1), "token_budget": os.environ.get("PS09_TOKEN_BUDGET", "65536"),
                     "wall_s": round(time.time() - t, 1), "when": time.strftime("%Y-%m-%d %H:%M")})
    keys = ["when", "stage", "name", "status", "passages_per_s", "seconds", "speedup", "min_cos", "violations",
            "live_p50_ms", "live_p99_ms", "live_min_cos", "setup_s", "token_budget", "wall_s"]
    f = res_dir / "ablation.csv"; new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow(r)
    md = ["| Stage | Passages/s | Speed-up | Min cos | Violations | Live p50 ms | Live p99 ms | Live min cos |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r.get('passages_per_s','')} | {r.get('speedup','')} | {r.get('min_cos','')} | "
                  f"{r.get('violations','')} | {r.get('live_p50_ms','')} | {r.get('live_p99_ms','')} | {r.get('live_min_cos','')} |"
                  if r["status"] == "ok" else f"| {r['name']} | {r['status']} | | | | | | |")
    open(res_dir / "ABLATION.md", "a").write(f"\n### Run {time.strftime('%Y-%m-%d %H:%M')}\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
