# Legacy TF-Encrypted 0.8.0 provenance

- Upstream: `https://github.com/tf-encrypted/tf-encrypted.git`
- Release/tag: `0.8.0`
- Commit: `51de98f9c2208d5d4bd716bf7316d3cd4e0bf434`
- Runtime: Python 3.6.13, TensorFlow 1.15.5, TF-Encrypted 0.8.0
- Installed package source comparison: no modified common source files relative
  to the upstream commit. Therefore no TFE source patch or Git bundle is needed.
- Custom ops are rebuilt with `scripts/new_build_tfe_custom_ops.sh`.

Current host custom-op checksums:

```text
d0522709c817c63e770327fc69e5dc4e422a9f3a7cd3e0af19b0a4375a485c73  aux_module_tf_1.15.5.so
7f3ab2d053761f1f60eb511547d0839285ffc341788f7f5b179c8e52dc24d340  secure_random_module_tf_1.15.5.so
```

The `.so` files are architecture-dependent backups in the large migration
archive, not Git assets. AutoDL should rebuild them against its own TensorFlow
installation and use the checksums only as provenance for the original host.

