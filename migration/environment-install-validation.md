# Clean-environment installation validation

No formal experiment is executed by this validation.

- YAML parsing and both requirements-file parsers: **PASS**.
- Modern Conda online dry-run: **not completed** because the source host could
  not reach `conda-forge/repodata.json` (`CondaHTTPError HTTP 000`).
- Modern Conda offline dry-run: solver started correctly but the local cache
  lacks NumPy 1.22.4 and Matplotlib 3.5.3. This is not a dependency conflict.
- Modern pip resolver dry-run: source-host pip predates the `--dry-run` option.
  Current-environment `pip check` only reports conflicts from an unrelated
  locally installed `cryptgpu` package, which is absent from `requirements.txt`.
- Legacy Conda offline dry-run: solver started correctly but the local cache
  lacks the old Python 3.6-era packages. AutoDL must use network/archived Conda
  channels or the frozen explicit snapshot.
- The original legacy environment runs NumPy 1.19.2, but TensorFlow 1.15.5
  declares `numpy<1.19`. The portable legacy specs therefore use NumPy 1.18.5;
  the exact original state remains recorded in the frozen snapshots.
- A true clean install could not be completed on this host without external
  Conda metadata. Target validation commands are included in
  `AUTODL_MIGRATION.md`; custom ops are rebuilt only after environment creation.
