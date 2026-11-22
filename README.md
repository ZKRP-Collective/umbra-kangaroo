# UmbraKangaroo

UmbraKangaroo is a research prototype of an interval-ECDLP solver for
secp256k1. Given `Q = kG` and a promise `k in [low, high]` (64-bit endpoints),
it recovers `k` with a parallel kangaroo walk on negation classes `{P, -P}`.

## Algorithm

- **Class walk.** The jump index comes from `x(P)` and the jump sign from the
  parity of `y(P)`, so each step is a function of the class `{P, -P}`.
- **Gated selection.** In `gated` mode, when the primary candidate `P ± J` has
  a poor 2-bit score of `x`, the other sign is also computed (one extra
  mul + sqr with the same inverse) and the lower score is kept. `lite` gates
  only on the worst score.
- **Shadow DPs.** `*-shadow` modes also submit the rejected candidate's `x` to
  the DP table.
- **Herds.** Tame walks start in a narrow band (`range/32`); wild walks cover
  the range plus an edge margin; jumps are even.
- **Batching and loops.** Fruitless loops are escaped via a 16-entry history.
  128 walkers share one Montgomery batch inversion per step.
- **Verification.** Every recovered scalar is checked by recomputing its point.

`gs_solver` is a reference: Gaudry–Schost with the negation map and
Galbraith–Ruprai set shapes (tame `[-N/2, N/2]`, wild `k' + [-N/4, N/4]`),
walks restarting at each DP, and 2-cycle escape.

## Build and run

```sh
make -C portable all test
make -C hip cpu-test                       # MI50 field leaf on a host stub
python3 experiments/test_experiments.py

portable/build/range_solver PUBKEY LOW_HEX HIGH_HEX [baseline|gated|lite|lite-shadow|full-shadow] [SEED_HEX]
portable/build/gs_solver    PUBKEY LOW_HEX HIGH_HEX DP_BITS [SEED_HEX]
```

Compile-time knobs: `RCK_DP_BITS`, `RCK_SCORE_LIMB`, `RCK_COUNT_MULS` (prints
the total number of field multiplications), `RCK_JUMP_COUNT`,
`RCK_WALKER_LIMIT`, plus the OpenMP, compact-DP and IFMA targets in
`portable/Makefile`. `experiments/hypothesis_bench.py` runs paired
comparisons on random targets.

## Results

`K = steps / sqrt(N)`. `mul/sqrtN` counts every field multiplication per
solve, including setup and inversions. All comparisons are paired: the same
random targets and seeds for every mode.

| Range | Targets | Mode | mean K | mul/step | mul/sqrtN |
|---|---:|---|---:|---:|---:|
| 2^32 | 100 | `baseline` | 1.40 | 9.81 | 13.8 |
| 2^32 | 100 | `gated` | 1.15 | 11.52 | 13.5 |
| 2^36 | 100 | `baseline` | 1.27 | 8.98 | 11.5 |
| 2^36 | 100 | `gated` | 1.03 | 10.54 | 11.1 |
| 2^40 | 60 | `baseline` | 1.22 | 8.72 | 10.7 |
| 2^40 | 60 | `lite` | 1.03 | 9.23 | 9.6 |
| 2^40 | 60 | `gated` | 0.95 | 10.22 | 9.9 |
| 2^40 | 60 | `full-shadow` | 0.85 | 10.23 | 8.9 |
| 2^44 | 30 | `baseline` | 1.34 | 8.64 | 11.7 |
| 2^44 | 30 | `gated` | 1.12 | 10.11 | 11.5 |

Findings:

- **Baseline.** `baseline` K is about 1.3–1.4, the usual level for a kangaroo
  with the negation map.
- **Gating.** Gating cuts steps by about 20%, but each step costs about 17%
  more multiplications. Total work `gated`/`baseline` at 2^40 is 0.88
  [0.69, 1.13] (dp 4) and 0.90 [0.71, 1.14] (dp 8), so there is no significant
  saving. A K below ~1.36 reflects counting steps while evaluating about 1.75
  candidate points per step.
- **Shadow DPs.** Shadow DPs only raise the effective DP density. At equal
  memory, `dp12+shadow` vs `dp11 gated` is 1.05 [0.98, 1.13].
- **Bit coupling.** Moving the score from `x.limb[0]` (shared with the jump
  index) to `x.limb[3]` has no significant effect.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
