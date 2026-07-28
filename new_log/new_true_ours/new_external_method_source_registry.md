# External method source registry

Status: `SOURCE_IDENTITY_AUDITED_NUMERIC_TABLE_PENDING_MAIN_RESULTS`

The old report's “PEA 2025 / about 84%” row is not source-safe and must not be reused. PEA is an IEEE S&P 2023 paper. Its original source reports the following CIFAR-10 results for SimCLR features and logistic regression, with five runs, epsilon 2, delta `1/(10n)`, and three parties:

- PEA-TF-Encrypted: `88.99 ± 0.08%`; more than 88% within 7 minutes in its LAN setting.
- PEA-Queqiao: `88.18 ± 0.33%`; more than 88% within 55 minutes in its LAN setting.

PEA's same table gives CaPC `86.70 ± 0.17%` at epsilon 2 and DDGauss `53.31 ± 2.52%` at epsilon 5. These are PEA-author reproductions using the official methods on PEA's extracted features. They are not native numbers reported by the CaPC or DDGauss papers. PEA states that DDGauss did not converge at epsilon 2. CaPC also protects a different PATE-style collaborative-labeling privacy scope. Consequently, these rows may be used only with explicit source and comparability footnotes, not as a strict ranking.

CryptGPU remains textual background only. PEA reports a non-private VGG-16 taking about 12 seconds per iteration and more than 16 hours to reach about 88% on CIFAR-10. The original CryptGPU paper still requires independent verification before final prose. No local speedup ratio may be computed from it.

Ours remains blank until the rebuilt protocol is frozen and all formal three-seed runs are complete. The rebuild has not read locked holdout or official test for tuning.

Primary sources:

- PEA: <https://arxiv.org/abs/2208.08662>, arXiv source SHA256 `2ea7323a0f674a15c0f55597fb0e4a5292b9dfb716b2152ba0a9acee8c42bdfe`.
- CaPC: <https://openreview.net/forum?id=h2EbJ4_wMVq>; official repository <https://github.com/cleverhans-lab/capc-iclr>, audited at `3024d6ae3b2e397692cf1784d86c30846a02221c`.
- DDGauss: <https://arxiv.org/abs/2102.06387>; official repository <https://github.com/google-research/federated>.

The machine-readable JSON contains the full privacy, MPC, hardware, provenance, and comparability fields.
