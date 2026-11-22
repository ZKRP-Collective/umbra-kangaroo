#!/usr/bin/env python3
"""Validate lazy P+J/P-J formulas on secp256k1 and count field work."""

from __future__ import annotations

import argparse
import dataclasses
import random
from collections.abc import Callable


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
Point = tuple[int, int] | None


@dataclasses.dataclass
class Ops:
    mul: int = 0
    sqr: int = 0
    inv: int = 0

    def __iadd__(self, other: "Ops") -> "Ops":
        self.mul += other.mul
        self.sqr += other.sqr
        self.inv += other.inv
        return self


def inverse(value: int) -> int:
    return pow(value % P, P - 2, P)


def point_add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = (3 * x1 * x1) * inverse(2 * y1) % P
    else:
        slope = (y1 - y2) * inverse(x1 - x2) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def point_neg(point: Point) -> Point:
    if point is None:
        return None
    return point[0], (-point[1]) % P


def scalar_mul(scalar: int, point: Point = G) -> Point:
    result = None
    addend = point
    scalar %= N
    while scalar:
        if scalar & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result


def counted_mul(left: int, right: int, ops: Ops) -> int:
    ops.mul += 1
    return left * right % P


def counted_sqr(value: int, ops: Ops) -> int:
    ops.sqr += 1
    return value * value % P


def counted_inv(value: int, ops: Ops) -> int:
    ops.inv += 1
    return inverse(value)


def finish_y(
    x1: int,
    y1: int,
    x3: int,
    slope: int,
    ops: Ops,
) -> int:
    return (counted_mul((x1 - x3) % P, slope, ops) - y1) % P


def primary_candidate(point: Point, jump: Point, ops: Ops) -> Point:
    """RCK baseline: choose the jump sign from current y parity."""
    assert point is not None and jump is not None
    x1, y1 = point
    x2, y2 = jump
    if y1 & 1:
        y2 = -y2 % P
    inv_dx = counted_inv((x1 - x2) % P, ops)
    slope = counted_mul((y1 - y2) % P, inv_dx, ops)
    x3 = (counted_sqr(slope, ops) - x1 - x2) % P
    y3 = finish_y(x1, y1, x3, slope, ops)
    return x3, y3


def lazy_dual_candidate(
    point: Point,
    jump: Point,
    should_evaluate: Callable[[int], bool],
    score: Callable[[int], int],
    ops: Ops,
) -> tuple[Point, bool, bool]:
    """Calculate primary x, optionally alternate x, then only selected y."""
    assert point is not None and jump is not None
    x1, y1 = point
    x2, jump_y = jump
    primary_jump_y = -jump_y % P if y1 & 1 else jump_y
    alternate_jump_y = -primary_jump_y % P

    inv_dx = counted_inv((x1 - x2) % P, ops)
    primary_slope = counted_mul(
        (y1 - primary_jump_y) % P, inv_dx, ops
    )
    primary_x = (counted_sqr(primary_slope, ops) - x1 - x2) % P

    evaluated = should_evaluate(score(primary_x))
    selected_alternate = False
    selected_x = primary_x
    selected_slope = primary_slope
    if evaluated:
        alternate_slope = counted_mul(
            (y1 - alternate_jump_y) % P, inv_dx, ops
        )
        alternate_x = (
            counted_sqr(alternate_slope, ops) - x1 - x2
        ) % P
        if score(alternate_x) < score(primary_x):
            selected_alternate = True
            selected_x = alternate_x
            selected_slope = alternate_slope

    selected_y = finish_y(x1, y1, selected_x, selected_slope, ops)
    return (selected_x, selected_y), evaluated, selected_alternate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    policies = {
        "lite": (
            lambda value: value == 3,
            lambda x: x & 3,
        ),
        "mid2": (
            lambda value: value >= 2,
            lambda x: x & 3,
        ),
        "full-gated": (
            lambda value: value != 0,
            lambda x: x & 3,
        ),
    }
    totals = {name: Ops() for name in policies}
    evaluations = {name: 0 for name in policies}
    selections = {name: 0 for name in policies}

    for _ in range(args.cases):
        point = scalar_mul(rng.randrange(1, N))
        jump = scalar_mul(rng.randrange(1, N))
        assert point is not None and jump is not None

        baseline_ops = Ops()
        baseline = primary_candidate(point, jump, baseline_ops)
        signed_jump = point_neg(jump) if point[1] & 1 else jump
        if baseline != point_add(point, signed_jump):
            raise AssertionError("baseline formula mismatch")
        if baseline_ops != Ops(mul=2, sqr=1, inv=1):
            raise AssertionError(f"unexpected baseline count: {baseline_ops}")

        for name, (gate, score) in policies.items():
            ops = Ops()
            selected, evaluated, selected_alternate = lazy_dual_candidate(
                point, jump, gate, score, ops
            )
            expected_jump = (
                point_neg(signed_jump)
                if selected_alternate
                else signed_jump
            )
            if selected != point_add(point, expected_jump):
                raise AssertionError(f"{name} formula mismatch")
            expected_ops = Ops(
                mul=2 + int(evaluated),
                sqr=1 + int(evaluated),
                inv=1,
            )
            if ops != expected_ops:
                raise AssertionError(
                    f"{name} operation mismatch: {ops} != {expected_ops}"
                )
            totals[name] += ops
            evaluations[name] += int(evaluated)
            selections[name] += int(selected_alternate)

    print("policy       eval-rate select-rate M/step S/step I/step")
    for name in policies:
        count = args.cases
        ops = totals[name]
        print(
            f"{name:<13s} "
            f"{evaluations[name] / count:9.4f} "
            f"{selections[name] / count:11.4f} "
            f"{ops.mul / count:6.4f} "
            f"{ops.sqr / count:6.4f} "
            f"{ops.inv / count:6.4f}"
        )
    print("secp256k1 lazy dual-candidate tests: PASS")


if __name__ == "__main__":
    main()
