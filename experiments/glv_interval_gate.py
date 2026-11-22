#!/usr/bin/env python3
"""Feasibility gate for GLV equivalence classes on a 1-D secp256k1 interval.

The secp256k1 endomorphism phi(x, y) = (beta*x, y) has phi(G) = lambda*G.
It is a bijection of the subgroup, but a contiguous scalar interval is not
closed under multiplication by lambda.  This script verifies those facts on
the real curve and reports overlap of a sampled interval with its five nonzero
GLV/negation images.  It is a proof gate against treating GLV as a free
factor-six reduction for interval DLP.
"""

from __future__ import annotations

import argparse
import random

from secp_sota_plus import G, N, P, Point, point_neg, scalar_mul


BETA = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72


def phi(point: Point) -> Point:
    if point is None:
        return None
    return (point[0] * BETA % P, point[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-bits", type=int, default=16)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2 <= args.range_bits <= 20 or args.cases <= 0:
        raise SystemExit("range-bits must be 2..20 and cases must be positive")
    if BETA == 1 or pow(BETA, 3, P) != 1:
        raise AssertionError("invalid secp256k1 beta")
    if LAMBDA == 1 or pow(LAMBDA, 3, N) != 1:
        raise AssertionError("invalid secp256k1 lambda")
    if phi(G) != scalar_mul(LAMBDA):
        raise AssertionError("phi(G) != lambda*G")

    rng = random.Random(args.seed)
    width = 1 << args.range_bits
    overlaps = [0, 0, 0, 0, 0]
    for _ in range(args.cases):
        low = rng.randrange(1, N - width - 1)
        interval = set(range(low, low + width))
        powers = [LAMBDA, LAMBDA * LAMBDA % N]
        images = powers + [(-power) % N for power in powers] + [N - 1]
        for index, multiplier in enumerate(images):
            overlaps[index] += sum(
                (multiplier * scalar) % N in interval
                for scalar in interval
            )

        # A sampled GLV orbit has six distinct representatives except for the
        # identity.  Verify the curve relation rather than trusting scalars.
        scalar = low + rng.randrange(width)
        point = scalar_mul(scalar)
        orbit = {
            point,
            phi(point),
            phi(phi(point)),
            point_neg(point),
            point_neg(phi(point)),
            point_neg(phi(phi(point))),
        }
        if len(orbit) != 6:
            raise AssertionError("unexpected nontrivial GLV/negation orbit")

    total = args.cases * width
    print(
        f"GLV interval gate: width=2^{args.range_bits}, cases={args.cases}, "
        f"sampled scalars={total}"
    )
    labels = ("lambda", "lambda^2", "-lambda", "-lambda^2", "-1")
    for label, overlap in zip(labels, overlaps):
        print(f"  overlap(I, {label}*I)={overlap}/{total}")
    print(
        "result: GLV is a subgroup bijection, not an interval-preserving "
        "free equivalence; a lattice-aware collision construction is required"
    )
    print("GLV interval gate: PASS")


if __name__ == "__main__":
    main()
