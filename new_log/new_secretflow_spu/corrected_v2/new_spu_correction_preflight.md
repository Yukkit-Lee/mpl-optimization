# SecretFlow-SPU corrected-v2 preflight

Status: `CORRECTION_REQUIRED`

The existing five nominal runs and their derived figures are preserved but
quarantined as development diagnostics.  They are not acceptable paper-formal
evidence because the formal runner reveals client-local Stage-B state, creates
all deterministic DP-noise shares on the driver, omits formal privacy
accounting, changes effective batch sizes after the final freeze, and does not
separate compilation/infeed/evaluation from online secure runtime.

The prior Tiny Full result is `UNRESOLVED_RUNTIME_CRASH`, not a proven cgroup
OOM.  The raw Ray error lists OOM, SIGSEGV, forced shutdown, and other worker
failures as alternatives, while no direct cgroup peak/event evidence was
recorded.

Corrected code/results will use the `corrected_v2` namespace and run IDs.  No
previous result, log, figure, TFE file, or checkpoint is overwritten.

Optimization target: match or exceed TFE utility and throughput where supported
by measured evidence.  Positive performance is not assumed from framework age
and will not override correctness, privacy, or frozen-target gates.
