#!/usr/bin/env python3
"""Held-out sweep for reusable multi-target tame-bank budgets."""

from __future__ import annotations

import argparse
import csv
import io
import math
import pathlib
import statistics
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_list(text: str, convert):
    return [convert(value) for value in text.split(",")]


def run_one(args: argparse.Namespace, tame_k: float, dp_bits: int, seed: int):
    command = [
        "python3",
        "experiments/multitarget_traps.py",
        "--range-bits", str(args.range_bits),
        "--targets", str(args.targets),
        "--walkers", str(args.walkers),
        "--dp-bits", str(dp_bits),
        "--tame-k", str(tame_k),
        "--wild-k", str(args.wild_k),
        "--seed", str(seed),
        "--table-seed", str(args.table_seed),
        "--oracle-seed", str(args.oracle_seed),
        "--csv",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if not rows:
        raise RuntimeError(completed.stderr.strip() or "tame sweep produced no CSV")
    final = rows[-1]
    return {
        "tame_k": tame_k,
        "dp_bits": dp_bits,
        "seed": seed,
        "records": int(final["tame_records"]),
        "online_k": float(final["online_mean_k"]),
        "online_p95_k": float(final["online_p95_k"]),
        "amortized_k": float(final["amortized_k"]),
        "failures": int(final["failures"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-bits", type=int, default=18)
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--wild-k", type=float, default=16.0)
    parser.add_argument("--tame-k", default="2,4,8")
    parser.add_argument("--dp-bits", default="3,4,5")
    parser.add_argument("--seeds", default="1,11,29")
    parser.add_argument("--table-seed", type=int, default=22)
    parser.add_argument("--oracle-seed", type=int, default=33)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tame_values = parse_list(args.tame_k, float)
    dp_values = parse_list(args.dp_bits, int)
    seeds = parse_list(args.seeds, int)
    if (args.targets <= 0 or args.walkers < 2 or args.wild_k <= 0.0 or
            not tame_values or not dp_values or not seeds):
        raise SystemExit("invalid sweep configuration")

    rows = [
        run_one(args, tame_k, dp_bits, seed)
        for tame_k in tame_values
        for dp_bits in dp_values
        for seed in seeds
    ]
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summaries = []
    bsgs_records = (math.isqrt(1 << args.range_bits) + 1) // 2 + 1
    for tame_k in tame_values:
        for dp_bits in dp_values:
            group = [
                row for row in rows
                if row["tame_k"] == tame_k and row["dp_bits"] == dp_bits
            ]
            summaries.append({
                "tame_k": tame_k,
                "dp_bits": dp_bits,
                "records": statistics.mean(row["records"] for row in group),
                "online_k": statistics.mean(row["online_k"] for row in group),
                "online_p95_k": statistics.mean(
                    row["online_p95_k"] for row in group
                ),
                "amortized_k": statistics.mean(
                    row["amortized_k"] for row in group
                ),
                "failures": sum(row["failures"] for row in group),
                "records_vs_bsgs": statistics.mean(
                    row["records"] for row in group
                ) / bsgs_records,
            })
    summaries.sort(key=lambda row: (row["failures"], row["amortized_k"]))
    print(
        "tame_k,dp_bits,mean_records,mean_online_k,mean_online_p95_k,"
        "mean_amortized_k,failures,records_vs_bsgs,fits_bsgs_records"
    )
    for row in summaries:
        print(
            f"{row['tame_k']:g},{row['dp_bits']},{row['records']:.2f},"
            f"{row['online_k']:.6f},{row['online_p95_k']:.6f},"
            f"{row['amortized_k']:.6f},"
            f"{row['failures']},{row['records_vs_bsgs']:.6f},"
            f"{int(row['records_vs_bsgs'] <= 1.0)}"
        )


if __name__ == "__main__":
    main()
