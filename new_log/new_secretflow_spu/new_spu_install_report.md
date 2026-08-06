# SecretFlow-SPU installation report

Status: `COMPLETE`

## Selected installation

The isolated environment is `/root/autodl-tmp/new_envs/spu_e2e` (environment
name `spu_e2e`) with Python 3.10.20. The installed paper backend is
SecretFlow 1.14.0b0 with its declared compatible SPU
0.9.4.dev20250618. `pip check` reports no broken requirements.

The current SecretFlow repository installation guide requires Python 3.10 and
recommends `pip install -U secretflow`. The current SPU repository supports
Ubuntu 22.04 with Python 3.10/3.11 and recommends `pip install spu`. The SPU
README additionally states that normal use should be through SecretFlow rather
than the demonstration-only bare distributed helper. This matches the paper
plan's requirement to use the SecretFlow SPU device API.

References inspected:

- https://github.com/secretflow/secretflow/blob/main/README.zh-CN.md
- https://github.com/secretflow/secretflow/blob/main/docs/getting_started/installation.md
- https://github.com/secretflow/spu
- https://github.com/secretflow/spu/blob/main/INSTALLATION.md
- https://secret-flow.antgroup.com/docs/spu/en/getting_started/install.html

## Source and proxy policy

Python itself was created from the Tsinghua Conda `pkgs/main` mirror with all
proxy variables removed. The initial package download used official PyPI and
the user-provided proxy at `127.0.0.1:9999`. After measuring slow official
large-wheel throughput, installation was resumed with the Alibaba PyPI mirror
as the primary index and official PyPI as fallback. `NO_PROXY` excluded the
Alibaba mirror from the proxy; official fallback traffic continued to use
port 9999. Alibaba did not publish the core SecretFlow/SPU packages, so those
official wheels necessarily came from PyPI.

The first pip attempt was intentionally interrupted during downloads to change
the index policy; it installed no SecretFlow dependencies. The second attempt
reused the pip cache and completed normally. Both logs are retained with their
SHA256 values in the environment manifest.

## Targeted source installation

The released SecretFlow 1.14 core imports an external `sfl.distributed`
strategy for simulation mode. Following the user's source-install preference,
the official `secretflow/sfl` repository was shallow-cloned and pinned at
`c383e40f665063d7f7d87e437a73e015f87c435c`, then built from source as
`sfl==1.0.0.dev20260807`. Its device scheduler requirements are pinned to
Ray 2.52.0 and SecretFlow-RayFed 0.2.1a2.

The source package was installed with `--no-deps` deliberately. Its monolithic
metadata includes TensorFlow, PyTorch, audio/vision, ONNX, XGBoost, and
`secretflow-lite==1.13.0b0`; those are unrelated to this SPU device pipeline,
consume several GiB, and the lite 1.13 package would conflict with the selected
SecretFlow 1.14 core namespace. The actual scheduler imports were diagnosed
incrementally; only Ray/RayFed and Ray Tune's small import-time dependencies
were added. Consequently the original core environment passed `pip check`,
while the optional full-SFL metadata check reports intentionally absent FL
algorithm extras. The real device smoke is the hard runtime gate.

Source-SFL exposes its official strategy classes but the pinned commit does not
register them as an import side effect. The isolated backend adapter therefore
uses SecretFlow's public `add_distribution_modes` hook to register the official
`SimulationStrategy`; neither installed source tree is patched.

## Version notes

- `ray==2.52.0`, `secretflow-rayfed==0.2.1a2`, and the pinned source-SFL build
  provide the SecretFlow device scheduler. They are not used by the bare-SPU
  quickstart gate.
- The exact transitive environment is frozen in `new_spu_pip_freeze.txt`.
- This environment is separate from `/root/autodl-tmp/new_envs/tfe`; no legacy
  TFE package or artifact was modified.
- Environment size after the targeted source extension is approximately 1.7
  GiB. Project-disk free space remains approximately 30 GiB.
