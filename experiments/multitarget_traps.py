#!/usr/bin/env python3
"""Finite-chain reference for a reusable multi-target kangaroo tame bank.

The group model uses exact integer generator coefficients and a deterministic
negation-compatible random oracle, as in walk_sim.cpp.  Tame chains are built
once for one interval/policy/jump table; each target then executes only wild
chains against that immutable distinguished-point bank.  This measures the
offline/online idea before a GPU-resident flat table is implemented.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys


MASK64 = (1 << 64) - 1
ORACLE_SALT = 0xD1B54A32D192ED03
DP_SALT = 0x94D049BB133111EB
ORIENT_SALT = 0x8538EC25BD456EA3


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


class Oracle:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def x_bits(self, scalar: int) -> int:
        return mix64(abs(scalar) ^ self.seed ^ ORACLE_SALT)

    def orientation(self, scalar: int) -> bool:
        bit = (mix64(abs(scalar) ^ self.seed ^ ORIENT_SALT) & 1) != 0
        return not bit if scalar < 0 else bit

    def distinguished(self, scalar: int, bits: int) -> bool:
        return (mix64(abs(scalar) ^ self.seed ^ DP_SALT) & ((1 << bits) - 1)) == 0


class Walker:
    def __init__(self, scalar: int, distance: int) -> None:
        self.scalar = scalar
        self.distance = distance


def make_jumps(root: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [2 * (4 * root + rng.randrange(4 * root)) for _ in range(count)]


def advance(walker: Walker, jumps: list[int], oracle: Oracle) -> None:
    jump = jumps[oracle.x_bits(walker.scalar) % len(jumps)]
    delta = -jump if oracle.orientation(walker.scalar) else jump
    walker.scalar += delta
    walker.distance += delta


def build_tame_bank(
    interval: int,
    root: int,
    walkers: int,
    dp_bits: int,
    steps: int,
    seed: int,
    jumps: list[int],
    oracle: Oracle,
) -> tuple[dict[int, int], int]:
    """Return canonical x-class -> tame distance and exact offline work."""
    rng = random.Random(seed)
    tame_range = max(2, interval // 32)
    tame_walkers = [Walker(rng.randrange(tame_range), 0) for _ in range(walkers)]
    for walker in tame_walkers:
        walker.distance = walker.scalar
    bank: dict[int, int] = {}
    for step in range(steps):
        walker = tame_walkers[step % walkers]
        advance(walker, jumps, oracle)
        if oracle.distinguished(walker.scalar, dp_bits):
            bank.setdefault(abs(walker.scalar), walker.distance)
    return bank, steps


def solve_target(
    secret: int,
    interval: int,
    walkers: int,
    dp_bits: int,
    max_steps: int,
    seed: int,
    jumps: list[int],
    oracle: Oracle,
    tame_bank: dict[int, int],
) -> tuple[int, bool]:
    """Run wild-only chains and recover only through an immutable tame bank."""
    rng = random.Random(seed)
    edge = max(2, interval // 32)
    shifted_secret = secret + edge
    wild_range = interval + 2 * edge
    wilds = []
    for _ in range(walkers):
        distance = rng.randrange(wild_range) & ~1
        wilds.append(Walker(distance - shifted_secret, distance))

    for step in range(max_steps):
        walker = wilds[step % walkers]
        advance(walker, jumps, oracle)
        if not oracle.distinguished(walker.scalar, dp_bits):
            continue
        tame_distance = tame_bank.get(abs(walker.scalar))
        if tame_distance is None:
            continue
        if (walker.distance - tame_distance == shifted_secret or
                walker.distance + tame_distance == shifted_secret):
            return step + 1, True
    return max_steps, False


def checkpoints(count: int) -> list[int]:
    values = []
    current = 1
    while current < count:
        values.append(current)
        current *= 2
    values.append(count)
    return values


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-bits", type=int, default=18)
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--dp-bits", type=int, default=4)
    parser.add_argument("--tame-k", type=float, default=4.0)
    parser.add_argument("--wild-k", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--table-seed", type=int, default=22)
    parser.add_argument("--oracle-seed", type=int, default=33)
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 8 <= args.range_bits <= 32:
        raise SystemExit("--range-bits must be between 8 and 32")
    if args.targets <= 0 or args.walkers < 2 or args.dp_bits > 24:
        raise SystemExit("invalid target, walker, or DP configuration")
    if args.tame_k <= 0.0 or args.wild_k <= 0.0:
        raise SystemExit("tame-k and wild-k must be positive")

    interval = 1 << args.range_bits
    root = math.isqrt(interval)
    jumps = make_jumps(root, 512, args.table_seed)
    oracle = Oracle(mix64(args.oracle_seed ^ 0xE7037ED1A0B428DB))
    tame_steps = math.ceil(args.tame_k * root)
    wild_steps = math.ceil(args.wild_k * root)
    bank, offline_steps = build_tame_bank(
        interval, root, args.walkers, args.dp_bits, tame_steps,
        args.seed, jumps, oracle,
    )
    if not bank:
        raise AssertionError("tame bank has no distinguished points")

    rng = random.Random(args.seed ^ 0x9E3779B9)
    online_steps = []
    failures = 0
    for target_index in range(args.targets):
        steps, solved = solve_target(
            rng.randrange(interval), interval, args.walkers, args.dp_bits,
            wild_steps, args.seed + target_index + 1, jumps, oracle, bank,
        )
        online_steps.append(steps)
        failures += not solved

    rows = []
    for count in checkpoints(args.targets):
        prefix = online_steps[:count]
        total_online = sum(prefix)
        rows.append({
            "targets": count,
            "tame_steps": offline_steps,
            "tame_records": len(bank),
            "online_mean_k": total_online / count / root,
            "online_p95_k": percentile(prefix, 0.95) / root,
            "amortized_k": (offline_steps + total_online) / count / root,
            "failures": failures if count == args.targets else "",
        })

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(
            f"reusable tame bank: range=2^{args.range_bits}, "
            f"targets={args.targets}, records={len(bank)}, "
            f"offline={offline_steps / root:.4f}*sqrt(N)"
        )
        for row in rows:
            print(
                f"  targets={row['targets']:>4}: "
                f"online={row['online_mean_k']:.4f}*sqrt(N), "
                f"p95={row['online_p95_k']:.4f}*sqrt(N), "
                f"amortized={row['amortized_k']:.4f}*sqrt(N)"
            )
        print(f"failures={failures}")
    if failures:
        raise SystemExit(f"reusable tame bank failures: {failures}")
    print("reusable tame bank tests: PASS", file=sys.stderr)


if __name__ == "__main__":
    main()
