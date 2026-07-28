# External method source registry

Status: `SOURCE_IDENTITY_AUDITED_NUMERIC_TABLE_PENDING_MAIN_RESULTS`

The old report's “PEA 2025 / about 84%” row is not source-safe and must not be reused. PEA is an IEEE S&P 2023 paper. Its original source reports the following CIFAR-10 results for SimCLR features and logistic regression, with five runs, epsilon 2, delta `1/(10n)`, and three parties:

- PEA-TF-Encrypted: `88.99 ± 0.08%`; more than 88% within 7 minutes in its LAN setting.
- PEA-Queqiao: `88.18 ± 0.33%`; more than 88% within 55 minutes in its LAN setting.

PEA's same table gives CaPC `86.70 ± 0.17%` at epsilon 2 and DDGauss `53.31 ± 2.52%` at epsilon 5. These are PEA-author reproductions using the official methods on PEA's extracted features. They are not native numbers reported by the CaPC or DDGauss papers. PEA states that DDGauss did not converge at epsilon 2. CaPC also protects a different PATE-style collaborative-labeling privacy scope. Consequently, these rows may be used only with explicit source and comparability footnotes, not as a strict ranking.

CaPC's original ICLR 2021 CIFAR-10 setup is itself not an absolute-accuracy counterpart to Ours: it uses 50 effective parties, three querying parties, VGG-5 local models, epsilon 20, Gaussian sigma 7, and about 550 queries per querying party. Across five runs it reports a mean local-model accuracy gain of `+4.09 percentage points`, rather than a single final absolute accuracy. The final table should therefore keep the original CaPC result and PEA's epsilon-2 reproduction as separate provenance layers.

CryptGPU remains textual background only. Its original IEEE S&P 2021 paper reports a CIFAR-10/VGG-16 private-training iteration taking `12.14 seconds` and communicating `7.55 GB` at batch size 32. This was measured on three AWS p3.2xlarge instances, each with one NVIDIA Tesla V100 16 GB, over a 1.25 GB/s, 0.2 ms LAN. The paper's separately reported CIFAR-10 test accuracy for its average-pooling VGG-16 is `83.17%`.

PEA states that a VGG-16 takes about 12 seconds per iteration and more than 16 hours to reach about 88%. The `12.14-second` iteration is consistent with CryptGPU's private-training table, but CryptGPU itself does not directly report an 88%-at-16-hours point. That statement must therefore remain explicitly attributed to PEA and cannot be presented as an original CryptGPU result. No local speedup ratio may be computed from either timing.

Ours remains blank until the rebuilt protocol is frozen and all formal three-seed runs are complete. The rebuild has not read locked holdout or official test for tuning.

Primary sources:

- PEA: <https://arxiv.org/abs/2208.08662>, arXiv source SHA256 `2ea7323a0f674a15c0f55597fb0e4a5292b9dfb716b2152ba0a9acee8c42bdfe`.
- CaPC: <https://openreview.net/forum?id=h2EbJ4_wMVq> and <https://arxiv.org/abs/2102.05188>; arXiv source SHA256 `3b00645d654ff88920a89edeed75db350ac9856f011ce9e12793b8ed07c9c15a`; official repository <https://github.com/cleverhans-lab/capc-iclr>, audited at `3024d6ae3b2e397692cf1784d86c30846a02221c`.
- DDGauss: <https://arxiv.org/abs/2102.06387>; official repository <https://github.com/google-research/federated>.
- CryptGPU: <https://arxiv.org/abs/2104.10949>, DOI `10.1109/SP40001.2021.00098`, arXiv source SHA256 `47bada653a5fa1cbe072ffcfc41fb26b5d7bd263520795116f35a04758c943e0`.

The machine-readable JSON contains the full privacy, MPC, hardware, provenance, and comparability fields.
