#!/usr/bin/env python3
"""Held-out parameter sweep for finite-chain kangaroo policies."""

from __future__ import annotations

import argparse
import csv
import io
import pathlib
import statistics
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ("default", 32, 1.0 / 3.0, 8.0, 8.0),
    ("jump12-tame64", 64, 0.30, 12.0, 8.0),
    ("jump12-tame32", 32, 0.366, 12.0, 8.0),
    ("jump4-tame32", 32, 0.30, 4.0, 4.0),
)


def run_one(binary, range_bits, trials, seed, name, divisor, fraction, jump_min, jump_width):
    command = [
        str(binary), "--range-bits", str(range_bits), "--trials", str(trials),
        "--threads", "4", "--max-k", "8", "--policy", "plus-full-gated",
        "--jump-min-root", str(jump_min), "--jump-width-root", str(jump_width),
        "--tame-divisor", str(divisor), "--tame-fraction", str(fraction),
        "--seed", str(seed), "--table-seed", "22", "--oracle-seed", "33",
        "--csv",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    row = next(csv.DictReader(io.StringIO(completed.stdout)))
    return {
        "config": name,
        "range_bits": range_bits,
        "seed": seed,
        "mean_k": float(row["mean_k"]),
        "stderr_k": float(row["stderr_k"]),
        "p95_k": float(row["p95_k"]),
        "failures": int(row["failures"]),
        "escape_rate": float(row["escape_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "experiments/walk_sim")
    parser.add_argument("--range-bits", default="32,36")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--seeds", default="11,17,29,43")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    ranges = [int(value) for value in args.range_bits.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.trials <= 0 or not ranges or not seeds:
        parser.error("trials, ranges, and seeds must be non-empty/positive")

    rows = []
    for range_bits in ranges:
        for name, divisor, fraction, jump_min, jump_width in DEFAULT_CONFIGS:
            for seed in seeds:
                rows.append(run_one(
                    args.binary, range_bits, args.trials, seed, name,
                    divisor, fraction, jump_min, jump_width,
                ))

    fields = list(rows[0])
    stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()

    for range_bits in ranges:
        print(f"range=2^{range_bits}", file=sys.stderr)
        for name, *_ in DEFAULT_CONFIGS:
            group = [row for row in rows
                     if row["range_bits"] == range_bits and row["config"] == name]
            values = [float(row["mean_k"]) for row in group]
            p95 = [float(row["p95_k"]) for row in group]
            failures = sum(int(row["failures"]) for row in group)
            se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            print(
                f"  {name:<18} mean_K={statistics.mean(values):.6f} "
                f"seed_SE={se:.6f} mean_p95={statistics.mean(p95):.6f} "
                f"failures={failures}", file=sys.stderr,
            )


if __name__ == "__main__":
    main()
