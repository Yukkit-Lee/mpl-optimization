# Full-trajectory SPU preflight (Phase G3)

Status: `GATES PASS, FROZEN`

## Gates summary

| Gate | Result |
|---|---|
| G0 synthetic ABY3 smoke | PASS (2 epochs + checkpoint eval + secret checkpoint round-trip, identical terminal eval) |
| G1 Gate A (no eval, 5 epochs) | PASS (0 Stage-B points, 0 client reveals, 10.36s Stage-B online) |
| G1 Gate B (eval enabled, 5 epochs) | PASS (4 Stage-B points, 80 aggregate-dev-logits reveals, 0 client reveals, dev acc 0.6474, eval overhead ~2.1s/checkpoint) |
| G1 OMP8/10 + chunk 250/500 | OMP8 frozen (OMP10 under 5% gate) |
| G2 Tiny exact-shape eval smoke | PASS (11.76s/checkpoint, RSS 0.67 GiB, 26 evals extrapolate to ~5 min < 3 h) |

## CPU freeze decision

OMP8 / num_cpus=30 / taskset 224-255 / chunk_rows 250. OMP10 was rejected:
CIFAR-10 only +2.8% faster (3.64 vs 3.75 s), Tiny -4.8% slower (12.32 vs
11.76 s); both under the 5% stability gate. chunk 500 rejected (4.24 vs 3.75 s).

## Eval overhead estimate (all six cells)

~94 s total Stage-B checkpoint-evaluation overhead, negligible vs ~14-18 h
Stage-B training.

## Hard gates all green

- no client-local reveal (0 everywhere so far);
- only aggregate development logits revealed with used_for_training=false;
- official test / locked holdout unread;
- conditional Stage-C DP only; Stage-B non-DP;
- per-party secret share checkpoints; resume preserves monotonic trace.
