#!/usr/bin/env python3
"""Compare interval-DLP method constants on one normalized cost frontier.

The output is deliberately arithmetic-only.  It answers whether a candidate
can beat Umbra/SOTA before implementation work by calculating how much faster
its group operation would need to be.  Pollard rho is normalized from the full
secp256k1 subgroup to the requested interval width.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-bits", default="60,80,100,120")
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--umbra-k", type=float, default=1.15)
    parser.add_argument("--hbm-gib", type=float, default=24.0)
    parser.add_argument("--record-bytes", type=int, default=32)
    parser.add_argument("--load-factor", type=float, default=0.70)
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def rows_for_range(
    bits: int,
    targets: int,
    umbra_k: float,
    max_bsgs_bits: float,
) -> list[dict[str, object]]:
    # Full-group Pollard rho needs sqrt(pi*q/2) operations.  Dividing by
    # sqrt(2^bits) expresses it in the same K*sqrt(interval) units.
    rho_k = math.sqrt(math.pi / 2.0) * 2.0 ** ((256 - bits) / 2.0)
    # Reusable negation-BSGS: about 0.5*sqrt(N) setup and 0.5*sqrt(N)
    # mean giant work.  Setup is shared by every target.
    reusable_bsgs_k = 0.5 / targets + 0.5
    methods = (
        ("umbra-sota-v2", umbra_k, "O(DP)", True),
        (
            "reusable-negation-bsgs",
            reusable_bsgs_k,
            "O(sqrt(N))",
            bits <= max_bsgs_bits,
        ),
        ("four-set-gaudry-schost", 1.661, "O(DP)", True),
        ("classic-two-kangaroo", 2.0, "O(DP)", True),
        ("full-group-pollard-rho", rho_k, "O(DP)", True),
    )
    return [
        {
            "range_bits": bits,
            "targets": targets,
            "method": name,
            "normalized_k": k,
            "work_vs_umbra": k / umbra_k,
            "step_speed_needed_to_tie": k / umbra_k,
            "memory": memory,
            "fits_memory": int(fits),
        }
        for name, k, memory, fits in methods
    ]


def main() -> None:
    args = parse_args()
    ranges = [int(value) for value in args.range_bits.split(",")]
    if (not ranges or any(bits <= 0 or bits >= 256 for bits in ranges) or
            args.targets <= 0 or args.umbra_k <= 0.0 or
            args.hbm_gib <= 0.0 or args.record_bytes <= 0 or
            not 0.0 < args.load_factor <= 1.0):
        raise SystemExit("invalid range, target count, or Umbra K")
    capacity = int(
        args.hbm_gib * (1 << 30) * args.load_factor / args.record_bytes
    )
    max_bsgs_bits = 2.0 * (math.log2(capacity) + 1.0)
    rows = [
        row
        for bits in ranges
        for row in rows_for_range(
            bits, args.targets, args.umbra_k, max_bsgs_bits
        )
    ]
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return

    print(
        f"method frontier: targets={args.targets}, Umbra K={args.umbra_k:g}, "
        f"full-BSGS ceiling≈2^{max_bsgs_bits:.2f}"
    )
    for bits in ranges:
        print(f"range=2^{bits}")
        for row in (item for item in rows if item["range_bits"] == bits):
            print(
                f"  {row['method']:<28} K={row['normalized_k']:.6g} "
                f"work={row['work_vs_umbra']:.6g}x Umbra "
                f"memory={row['memory']} fits={bool(row['fits_memory'])}"
            )


if __name__ == "__main__":
    main()
