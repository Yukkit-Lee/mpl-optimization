# Secure Heterogeneity Trigger Plaintext Smoke Report

## S0 crossover

| Dataset | IID B0→B1 | Mild B0→B1 | Severe B0→B1 |
|---|---:|---:|---:|
| cifar10 | 22.70→57.68% | 22.02→55.76% | 18.84→57.70% |
| cifar100 | 10.24→22.12% | 9.06→20.34% | 8.08→17.62% |

The required crossover is absent: B1 is better than B0 in every dataset/distribution. H1-H3 are monotonic, but heterogeneity does not indicate whether anchor CE is beneficial because anchor CE is beneficial everywhere.

Best registered extreme-routing candidate: `H1_mean_tv` at threshold `0.21869851875313867`; mean/max oracle regret `7.81/34.98pp`. It therefore fails the required 0.5/1.0pp regret gates.
