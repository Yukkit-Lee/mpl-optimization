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

## Why source installation was not selected

SecretFlow's source option builds only the SecretFlow Python wheel from the
moving `main` branch. Installing that wheel still resolves the same binary
SPU, JAXLIB, PyArrow, and related dependencies, which were the actual download
cost. The repository also warns that non-release builds are not production
releases. Since the published wheel installed consistently and `pip check`
passed, replacing it with `main` would reduce version reproducibility without
removing the binary dependency downloads. Source installation remains a
fallback only if the required SecretFlow device smoke exposes a release bug.

## Version notes

- `ray` is `NOT_INSTALLED`: SecretFlow 1.14.0b0 does not declare Ray as a
  dependency. This is recorded explicitly rather than silently substituting a
  package.
- The exact transitive environment is frozen in `new_spu_pip_freeze.txt`.
- This environment is separate from `/root/autodl-tmp/new_envs/tfe`; no legacy
  TFE package or artifact was modified.
- Environment size after installation is approximately 1.4 GiB. Project-disk
  free space remains approximately 31 GiB.
