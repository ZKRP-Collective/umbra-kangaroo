#!/usr/bin/env python3
"""Reproducible real-curve A/B benchmark for CPU hypothesis H4."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import re
import shutil
import math
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = {
    36: (
        "03019c11b6a1009fb0f3ca0b7e34e9aecef2cbe9852bad7b3171bacd6a34a33ad9",
        0x8BADF0055,
    ),
    40: (
        "03353e9784f0ef932fa0d176262b1c6e9673d0715b7f9ed03fba057562d68507ec",
        0x8BADF00D55,
    ),
}
VARIANTS = (
    ("baseline", "openmp-product-tree-compact"),
    ("h4-gate1", "openmp-overbooked"),
    ("h4-gate2", "openmp-overbooked-gate2"),
    ("h4-gate3", "openmp-overbooked-gate3"),
)
METRIC = re.compile(r"([A-Za-z-]+)=([0-9a-f.]+)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, (len(ordered) * int(fraction * 100) + 99) // 100 - 1)
    return ordered[min(index, len(ordered) - 1)]


def build_variants(destination: pathlib.Path) -> dict[str, pathlib.Path]:
    binaries: dict[str, pathlib.Path] = {}
    for name, target in VARIANTS:
        subprocess.run(
            ["make", "-C", "portable", target],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        binary = destination / name
        shutil.copy2(ROOT / "portable/build/range_solver", binary)
        binaries[name] = binary
    return binaries


def run_one(
    binary: pathlib.Path,
    public_key: str,
    range_bits: int,
    seed: int,
    environment: dict[str, str],
) -> dict[str, str | int | float]:
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(binary),
            public_key,
            "0",
            format((1 << range_bits) - 1, "x"),
            "full-shadow",
            format(seed, "x"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"solver failed for {binary.name}, 2^{range_bits}, seed {seed}:\n"
            f"{result.stdout}{result.stderr}"
        )
    metrics = dict(METRIC.findall(result.stdout))
    expected_private = f"{TARGETS[range_bits][1]:064x}"
    if metrics.get("private") != expected_private:
        raise AssertionError(
            f"wrong private scalar for {binary.name}, seed {seed}: "
            f"{metrics.get('private')} != {expected_private}"
        )
    return {
        "range_bits": range_bits,
        "seed": seed,
        "variant": binary.name,
        "wall_seconds": elapsed,
        "steps": int(metrics["steps"]),
        "k": float(metrics["K"]),
        "alt_eval": int(metrics["alt-eval"]),
        "second_jump_eval": int(metrics["second-jump-eval"]),
        "second_jump_selected": int(metrics["second-jump-selected"]),
        "loop_escapes": int(metrics["loop-escapes"]),
    }


def summarize(rows: list[dict[str, str | int | float]]) -> None:
    print(
        "range variant              mean_s   speedup   mean_K     SE_K    p95_K  "
        "second_eval second_select",
        file=sys.stderr,
    )
    for range_bits in sorted({int(row["range_bits"]) for row in rows}):
        selected = [row for row in rows if row["range_bits"] == range_bits]
        baseline = [
            float(row["wall_seconds"])
            for row in selected
            if row["variant"] == "baseline"
        ]
        baseline_mean = statistics.mean(baseline)
        for name, _ in VARIANTS:
            group = [row for row in selected if row["variant"] == name]
            walls = [float(row["wall_seconds"]) for row in group]
            ks = [float(row["k"]) for row in group]
            standard_error = (
                statistics.stdev(ks) / math.sqrt(len(ks))
                if len(ks) > 1
                else 0.0
            )
            steps = sum(int(row["steps"]) for row in group)
            second_eval = sum(int(row["second_jump_eval"]) for row in group)
            second_selected = sum(
                int(row["second_jump_selected"]) for row in group
            )
            print(
                f"2^{range_bits:<2} {name:<20} "
                f"{statistics.mean(walls):7.4f} "
                f"{baseline_mean / statistics.mean(walls):8.3f} "
                f"{statistics.mean(ks):8.4f} {standard_error:8.4f} "
                f"{percentile(ks, 0.95):8.4f} "
                f"{second_eval / steps:11.3%} "
                f"{second_selected / steps:13.3%}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--range-bits", default="36,40")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    ranges = [int(value) for value in args.range_bits.split(",")]
    unsupported = sorted(set(ranges) - TARGETS.keys())
    if unsupported:
        parser.error(f"fixed verified targets exist only for 36,40; got {unsupported}")
    if args.seeds < 1 or args.threads < 1:
        parser.error("--seeds and --threads must be positive")

    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS=str(args.threads),
        OMP_PROC_BIND="true",
        OMP_PLACES="cores",
    )
    rows: list[dict[str, str | int | float]] = []
    with tempfile.TemporaryDirectory(prefix="rck-h4-") as temporary:
        binaries = build_variants(pathlib.Path(temporary))
        names = [name for name, _ in VARIANTS]
        for range_bits in ranges:
            public_key, _ = TARGETS[range_bits]
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                rotation = (seed - args.seed_start) % len(names)
                order = names[rotation:] + names[:rotation]
                for name in order:
                    rows.append(
                        run_one(
                            binaries[name],
                            public_key,
                            range_bits,
                            seed,
                            environment,
                        )
                    )

    fieldnames = list(rows[0])
    if args.output:
        stream = args.output.open("w", newline="", encoding="utf-8")
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    summarize(rows)


if __name__ == "__main__":
    main()
