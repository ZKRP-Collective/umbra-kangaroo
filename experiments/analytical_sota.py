#!/usr/bin/env python3
"""Idealized SOTA-v2 collision model.

This is deliberately independent of CUDA/HIP and of wall-clock cost.  It
reproduces the analytical alpha result and explores the tame/wild allocation
before those policies are tested with finite deterministic walks.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable


def overlap(alpha: float, u: float) -> float:
    """Scaled overlap h(u) for 0 <= u <= 1/2."""
    boundary = (1.0 - alpha) / 2.0
    if u <= boundary:
        return alpha
    return (1.0 + alpha) / 2.0 - u


def collision_weight(alpha: float, tame_fraction: float, u: float) -> float:
    """Coefficient A in survival exp(-t^2 A/N), before negation."""
    tame_wild = (
        tame_fraction
        * (1.0 - tame_fraction)
        * overlap(alpha, u)
        / alpha
    )
    wild_wild = (
        (1.0 - tame_fraction) ** 2
        * (1.0 - 2.0 * u)
        / 2.0
    )
    return tame_wild + wild_wild


def simpson_integral(
    fn: Callable[[float], float], start: float, stop: float, intervals: int
) -> float:
    """Composite Simpson integral with an even interval count."""
    if intervals <= 0 or intervals % 2:
        raise ValueError("interval count must be a positive even number")
    step = (stop - start) / intervals
    total = fn(start) + fn(stop)
    total += 4.0 * sum(fn(start + step * i) for i in range(1, intervals, 2))
    total += 2.0 * sum(fn(start + step * i) for i in range(2, intervals, 2))
    return total * step / 3.0


def k_numeric(
    alpha: float, tame_fraction: float = 1.0 / 3.0, intervals: int = 200_000
) -> float:
    """Expected K with the negation map for a uniform interval secret."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 < tame_fraction < 1.0:
        raise ValueError("tame fraction must be in (0, 1)")

    integral = simpson_integral(
        lambda u: 1.0 / math.sqrt(collision_weight(alpha, tame_fraction, u)),
        0.0,
        0.5,
        intervals,
    )
    return math.sqrt(math.pi / 2.0) * integral


def k_closed(alpha: float) -> float:
    """Closed-form three-set result with the negation map."""
    bracket = (
        1.0
        - math.sqrt((1.0 + alpha) / 2.0)
        + alpha
        / (2.0 * alpha + 1.0)
        * (math.sqrt(2.0 * (1.0 + alpha)) - 1.0)
    )
    return 3.0 * math.sqrt(math.pi / 2.0) * bracket


def golden_minimize(
    fn: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Minimize a unimodal function on a bounded interval."""
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - inverse_phi * (upper - lower)
    right = lower + inverse_phi * (upper - lower)
    f_left = fn(left)
    f_right = fn(right)

    while upper - lower > tolerance:
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper - inverse_phi * (upper - lower)
            f_left = fn(left)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + inverse_phi * (upper - lower)
            f_right = fn(right)

    optimum = (lower + upper) / 2.0
    return optimum, fn(optimum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intervals",
        type=int,
        default=20_000,
        help="even Simpson interval count (default: 20000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.intervals <= 0 or args.intervals % 2:
        raise SystemExit("--intervals must be a positive even number")

    print("alpha       K(closed)   K(numeric)  optimum tame  optimum K")
    for denominator in (4, 8, 16, 32, 64, 128):
        alpha = 1.0 / denominator
        optimum_tame, optimum_k = golden_minimize(
            lambda fraction: k_numeric(alpha, fraction, args.intervals),
            0.05,
            0.80,
        )
        print(
            f"1/{denominator:<7d} "
            f"{k_closed(alpha):10.6f} "
            f"{k_numeric(alpha, intervals=args.intervals):11.6f} "
            f"{optimum_tame:13.6f} "
            f"{optimum_k:10.6f}"
        )

    for alpha in (1.0 / 8.0, 1.0 / 64.0):
        error = abs(k_closed(alpha) - k_numeric(alpha, intervals=args.intervals))
        if error > 2e-6:
            raise SystemExit(
                f"closed/numeric mismatch at alpha={alpha}: error={error}"
            )


if __name__ == "__main__":
    main()
