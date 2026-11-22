# UmbraKangaroo

An experimental interval-ECDLP solver for secp256k1. Given `Q = kG` and a
promise `k in [low, high]`, it recovers `k` with a parallel kangaroo walk on
negation classes `{P, -P}`. Research prototype: ranges are limited to 64-bit
endpoints.

## Algorithm

- **Class walk.** The jump index comes from `x(P)`. The sign of the jump comes
  from the parity of `y(P)`, so the walk is a function of the class `{P, -P}`.
- **Gated selection.** In `gated` mode, if the primary candidate `P ± J` has a
  poor 2-bit score of `x`, the other sign is also computed (one extra mul + sqr
  with the same inverse) and the lower score is kept. `lite` gates only on the
  worst score.
- **Shadow DPs.** `*-shadow` modes also submit the rejected candidate's `x` to
  the DP table.
- **Herds.** Tame walks start in a narrow band (`range/32`). Wild walks cover
  the range plus an edge margin. Jumps are even.
- **Batching and loops.** Fruitless loops are escaped through a 16-entry
  history. 128 walkers share one Montgomery batch inversion per step.
- **Verification.** Every recovered scalar is checked by recomputing its point.

`gs_solver` is the external reference: Gaudry–Schost with the negation map and
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

Compile-time knobs: `RCK_DP_BITS`, `RCK_SCORE_LIMB`, `RCK_JUMP_COUNT`,
`RCK_WALKER_LIMIT`, plus the OpenMP, compact-DP and IFMA targets in
`portable/Makefile`. `experiments/hypothesis_bench.py` runs paired
comparisons on random targets.

## Results (2^40, 60 random targets, same seeds per mode)

`K = steps / sqrt(N)`, mean ± s.e.

| Solver | dp bits | mean K | median K |
|---|---|---:|---:|
| GS + negation (reference) | 4 | 1.28 ± 0.08 | 1.23 |
| GS + negation (reference) | 8 | 1.77 ± 0.13 | 1.69 |
| Umbra `baseline` | 4 | 1.22 ± 0.09 | 1.04 |
| Umbra `gated` | 4 | 0.95 ± 0.08 | 0.83 |
| Umbra `full-shadow` | 4 | 0.85 ± 0.07 | 0.79 |
| Umbra `gated` | 8 | 0.98 ± 0.08 | 0.88 |
| Umbra `gated` | 12 | 1.30 ± 0.09 | 1.24 |

Paired findings (95% CI):

- **Gating.** `gated` vs `baseline` K ratio is 0.74–0.80 across dp 4/8/12;
  every CI excludes 1. No measurable per-step cost increase (±10%).
- **Shadow DPs.** Shadow DPs help (ratio 0.91–0.98), but only by raising
  effective DP density along merged paths. At equal memory, `dp12+shadow` vs
  `dp11 gated` is 1.05 [0.98, 1.13]: no gain beyond lowering `dp_bits`.
- **Bit coupling.** Moving the score from `x.limb[0]` (shared with the jump
  index) to `x.limb[3]` changes nothing significant (1.07, CI ≈ [0.85, 1.37]).
- **Escape accounting.** Escape jumps (0.8–1.6% of steps) are excluded from
  `steps`; including them adds 0.01–0.02 to K.

## Limitations and open questions

- **Below the known constant.** Umbra `gated` reaches K below the commonly
  cited ~1.36√N for interval DLP with negation. This needs an independent audit
  of the step accounting before any claim of improvement.
- **Reference speed.** `gs_solver` restarts are not batched. Compare K, not
  wall time.
- **64-bit ceiling.** Ranges above 2^64 are not supported, so nothing here is
  validated at 2^80+.
- **Single host.** All results are from one CPU; the MI50 path is a field
  leaf only.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
