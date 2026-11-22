#!/usr/bin/env python3
"""Dependency-free regression tests for the algorithm experiments."""

from __future__ import annotations

import csv
import io
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> None:
    hip_leaf = run("make", "-C", "hip", "cpu-test")
    if "gfx906 field leaf host-oracle tests: PASS" not in hip_leaf.stdout:
        raise AssertionError("gfx906 field leaf oracle test did not pass")

    run("make", "-C", "portable", "all")
    analytical = run(
        "python3",
        "experiments/analytical_sota.py",
        "--intervals",
        "20000",
    )
    if "1/64" not in analytical.stdout:
        raise AssertionError("analytical alpha sweep did not run")

    secp = run(
        "python3",
        "experiments/secp_sota_plus.py",
        "--cases",
        "64",
        "--seed",
        "1",
    )
    if "tests: PASS" not in secp.stdout:
        raise AssertionError("secp256k1 lazy-candidate tests did not pass")

    tame_bank = run(
        "python3",
        "experiments/multitarget_traps.py",
        "--range-bits",
        "14",
        "--targets",
        "16",
        "--walkers",
        "64",
        "--dp-bits",
        "3",
        "--tame-k",
        "8",
        "--wild-k",
        "16",
        "--csv",
    )
    tame_rows = list(csv.DictReader(io.StringIO(tame_bank.stdout)))
    if tame_rows[-1]["failures"] != "0":
        raise AssertionError("reusable tame bank did not solve every target")
    if not float(tame_rows[-1]["amortized_k"]) < float(
            tame_rows[0]["amortized_k"]):
        raise AssertionError("tame-bank setup was not amortized across targets")

    glv = run(
        "python3",
        "experiments/glv_interval_gate.py",
        "--range-bits",
        "12",
        "--cases",
        "4",
    )
    if "GLV interval gate: PASS" not in glv.stdout:
        raise AssertionError("GLV interval gate did not pass")

    frontier = run(
        "python3",
        "experiments/method_frontier.py",
        "--range-bits",
        "60,80",
        "--targets",
        "64",
        "--csv",
    )
    frontier_rows = list(csv.DictReader(io.StringIO(frontier.stdout)))
    by_key = {
        (int(row["range_bits"]), row["method"]): row
        for row in frontier_rows
    }
    if by_key[(60, "reusable-negation-bsgs")]["fits_memory"] != "1":
        raise AssertionError("2^60 BSGS should fit the 24 GiB model")
    if by_key[(80, "reusable-negation-bsgs")]["fits_memory"] != "0":
        raise AssertionError("2^80 BSGS should exceed the 24 GiB model")
    if not float(
            by_key[(80, "four-set-gaudry-schost")]["work_vs_umbra"]
            ) > 1.4:
        raise AssertionError("Gaudry-Schost frontier gate is inconsistent")

    trap_sweep = run(
        "python3",
        "experiments/multitarget_trap_sweep.py",
        "--range-bits", "14",
        "--targets", "16",
        "--walkers", "64",
        "--tame-k", "4,8",
        "--dp-bits", "2,3",
        "--seeds", "1,11",
    )
    trap_rows = list(csv.DictReader(io.StringIO(trap_sweep.stdout)))
    if len(trap_rows) != 4:
        raise AssertionError("reusable tame sweep matrix is incomplete")
    if any(int(row["failures"]) != 0 for row in trap_rows):
        raise AssertionError("reusable tame sweep lost a target")
    if not any(row["fits_bsgs_records"] == "1" for row in trap_rows):
        raise AssertionError("tame sweep did not expose equal-memory candidates")

    with tempfile.TemporaryDirectory(prefix="rck-walk-test-") as temp:
        binary = pathlib.Path(temp) / "walk_sim"
        run(
            "c++",
            "-O2",
            "-std=c++20",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "experiments/walk_sim.cpp",
            "-o",
            str(binary),
        )
        simulation = run(
            str(binary),
            "--range-bits",
            "28",
            "--trials",
            "64",
            "--threads",
            "4",
            "--max-k",
            "8",
            "--policy",
            "baseline,plus-full,plus-full-gated",
            "--csv",
        )

    rows = {
        row["policy"]: row
        for row in csv.DictReader(io.StringIO(simulation.stdout))
    }
    if set(rows) != {"baseline", "plus-full", "plus-full-gated"}:
        raise AssertionError(f"unexpected policies: {sorted(rows)}")
    if any(int(row["failures"]) != 0 for row in rows.values()):
        raise AssertionError("a finite-chain solve failed")

    full = rows["plus-full"]
    gated = rows["plus-full-gated"]
    for field in ("mean_k", "median_k", "p95_k", "p99_k", "dp_per_trial"):
        if full[field] != gated[field]:
            raise AssertionError(
                f"gated full changed the trajectory field {field}: "
                f"{full[field]} != {gated[field]}"
            )
    if not float(gated["alt_eval_rate"]) < float(full["alt_eval_rate"]):
        raise AssertionError("gating did not reduce alternate evaluations")

    print("experiment regression tests: PASS")


if __name__ == "__main__":
    main()
