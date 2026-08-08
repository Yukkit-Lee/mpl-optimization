# SecretFlow-SPU corrected-v2 end-to-end report (seed42)

Status: 6/6 COMPLETE

Backend: SecretFlow-SPU device API, ABY3, FM128, fxp16. All six cells
passed security/DP-accountant/reveal/freeze/timing/test-holdout gates.

| Dataset | Method | SPU dev acc | TFE dev acc | Δ | StageB online (s) | StageC online (s) | Total online (s) | Raw wall (s) | Mem peak (GiB) | ε | T@80 (s) | T@90 (s) | T@95 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | True Ours | 0.6864 | 0.6886 | -0.0022 | 110.4 | 25.5 | 135.9 | 213.3 | 13.4 | 3.0 | 113.6 | 113.6 | Not Reached |
| cifar10 | Full Data | 0.7254 | 0.7300 | -0.0046 | 1060.6 | 25.8 | 1086.4 | 2622.3 | 14.9 | 3.0 | 1063.8 | 1063.8 | 1063.8 |
| cifar100 | True Ours | 0.4766 | 0.4866 | -0.0100 | 2565.8 | 8.3 | 2574.1 | 3402.1 | 14.4 | 3.0 | 2570.0 | 2570.0 | 2570.0 |
| cifar100 | Full Data | 0.4252 | 0.4340 | -0.0088 | 6167.6 | 9.0 | 6176.6 | 8384.3 | 17.5 | 3.0 | 6172.2 | 6172.2 | 6172.2 |
| tiny | True Ours | 0.3084 | 0.3141 | -0.0057 | 6061.0 | 223.6 | 6284.7 | 8302.6 | 22.8 | 3.0 | 6067.6 | 6067.6 | 6067.6 |
| tiny | Full Data | 0.2768 | 0.2720 | +0.0048 | 21148.7 | 224.0 | 21372.7 | 26070.6 | 23.3 | 3.0 | 21154.7 | 21154.7 | 21154.7 |

## Main-trend reproduction

- CIFAR-10: Ours reaches T@90 9.4x faster than Full (113.6s vs 1063.8s).
- CIFAR-100: Ours reaches T@90 2.4x faster (2570.0s vs 6172.2s).
- Tiny: Ours reaches T@90 3.5x faster (6067.6s vs 21154.7s).

## Honest boundaries

- All accuracy is development-validation; official test / locked holdout unread.
- epsilon=3 is conditional Stage C DP only; Stage-B initializer is data-dependent non-DP.
- cgroup CPU utilization is diagnostic only (shared user Claude processes on the container).
- Tiny Full SPU cell now COMPLETE (old NEGATIVE_STOP resolved by chunked graph).
- Communication N/A.

