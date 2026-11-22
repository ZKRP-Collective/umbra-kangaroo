"""Paired real-curve benchmark of range_solver/gs_solver variants.

usage: python3 -I experiments/hypothesis_bench.py REPO_ROOT BITS TARGETS OUT.csv BIN:MODE,...
Binaries are read from REPO_ROOT/bins/. For gs_solver binaries, MODE is DP_BITS.
"""
import csv, random, re, subprocess, sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, sys.argv[1] + "/experiments")
from secp_sota_plus import G, scalar_mul

ROOT = sys.argv[1]
BITS = int(sys.argv[2]); SEEDS = int(sys.argv[3]); out = sys.argv[4]
configs = [c.split(":") for c in sys.argv[5].split(",")]  # bin:mode
rng = random.Random(BITS * 7919)
targets = []
for _ in range(SEEDS):
    k = rng.randrange(1, 1 << BITS)
    x, y = scalar_mul(k, G)
    targets.append((k, "%02x%064x" % (2 + (y & 1), x), rng.getrandbits(48)))
M = re.compile(r"([A-Za-z-]+)=([0-9a-fA-F.x]+)")

def run(job):
    b, mode, (k, pub, seed) = job
    t = time.perf_counter()
    r = subprocess.run([f"{ROOT}/bins/{b}", pub, "0", "%x" % ((1 << BITS) - 1), mode, "%x" % seed],
                       capture_output=True, text=True)
    dt = time.perf_counter() - t
    m = dict(M.findall(r.stdout))
    ok = m.get("private") == "%064x" % k
    return dict(bin=b, mode=mode, seed=seed, ok=int(ok), steps=int(m.get("steps", 0)),
                K=float(m.get("K", 0)), alt=int(m.get("alt-eval", 0)), shadow=int(m.get("shadow", 0)), esc=int(m.get("loop-escapes", 0)), dp=int(m.get("dp", 0)), wall=dt)

jobs = [(b, mo, t) for b, mo in configs for t in targets]
with ThreadPoolExecutor(int(__import__('os').environ.get('J','15'))) as ex:
    rows = list(ex.map(run, jobs))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
for b, mo in configs:
    rs = [r for r in rows if r["bin"] == b and r["mode"] == mo]
    ks = [r["K"] for r in rs if r["ok"]]
    ws = [r["wall"] for r in rs if r["ok"]]
    se = st.stdev(ks) / len(ks) ** 0.5 if len(ks) > 1 else 0
    print(f"{b:12} {mo:12} solved={sum(r['ok'] for r in rs)}/{len(rs)} meanK={st.mean(ks):.3f}±{se:.3f} "
          f"medK={st.median(ks):.3f} alt/step={st.mean(r['alt']/max(r['steps'],1) for r in rs):.2f} esc/step={st.mean(r['esc']/max(r['steps'],1) for r in rs):.4f} Kfull={st.mean((r['steps']+r['esc'])/2**(BITS/2) for r in rs if r['ok']):.3f} ns/step={1e9*sum(ws)/sum(r['steps']+r['esc'] for r in rs if r['ok']):.0f} dp+shadow={st.mean(r['dp']+r['shadow'] for r in rs):.0f}")
