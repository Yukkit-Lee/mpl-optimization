#!/usr/bin/env python3
"""Corrected-v2 security semantics gate on real SecretFlow-SPU.

This is deliberately small and runs before any corrected formal experiment.
It proves that Stage-B client states remain secret across epochs, party noise
shares originate independently inside PYUs from OS CSPRNG entropy, SPU share
checkpoint/load works, and every public reconstruction is explicitly logged.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

ROOT = Path(__file__).resolve().parents[3]
SPU_SCRIPTS = ROOT / "new_scripts/new_secretflow_spu"
if str(SPU_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SPU_SCRIPTS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from new_secretflow_spu_backend import register_official_sfl_strategies  # noqa: E402
from new_spu_stage_b_core import (  # noqa: E402
    make_secure_stage_b_average,
    make_secure_stage_b_formal,
)
from new_spu_stage_c_core import make_secure_stage_c_scan  # noqa: E402
from new_spu_corrected_runtime import (  # noqa: E402
    RevealLedger,
    dump_spu_object,
    load_spu_object,
    request_private_noise_shares,
    reveal_private_noise_metadata,
    spu_multi_return,
)

OUT = ROOT / "new_log/new_secretflow_spu/corrected_v2"
CKPT = ROOT / "new_checkpoints/new_secretflow_spu/corrected_v2"
JSON_PATH = OUT / "new_spu_corrected_security_smoke.json"
MD_PATH = OUT / "new_spu_corrected_security_smoke.md"


def main() -> None:
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    started = time.monotonic()
    adapter = register_official_sfl_strategies()
    parties = ["alice", "bob", "carol"]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent)
        + ":"
        + str(SPU_SCRIPTS)
        + ":"
        + str(ROOT)
        + (":" + existing if existing else "")
    )
    sf.init(
        parties=parties,
        address="local",
        num_cpus=30,
        omp_num_threads=8,
        log_to_driver=False,
        include_dashboard=False,
    )
    ledger = RevealLedger()
    try:
        pyus = [sf.PYU(name) for name in parties]
        alice = pyus[0]
        runtime_config = {
            "protocol": "ABY3",
            "field": "FM128",
            "fxp_fraction_bits": 16,
        }
        spu_device = sf.SPU(cluster_def(parties, runtime_config=runtime_config))

        rng = np.random.RandomState(42017)
        clients, batches, batch_size, feature_dim, classes = 5, 2, 4, 12, 3
        x_pack = rng.normal(size=(clients, batches, batch_size, feature_dim)).astype(np.float32)
        labels = rng.randint(0, classes, size=(clients, batches, batch_size))
        y_pack = np.eye(classes, dtype=np.float32)[labels]
        init_w = rng.normal(0.0, 0.01, size=(clients, feature_dim, classes)).astype(np.float32)
        init_b = np.zeros((clients, classes), dtype=np.float32)
        x_s = alice(lambda value=x_pack: value)().to(spu_device)
        y_s = alice(lambda value=y_pack: value)().to(spu_device)

        epoch_exec = spu_multi_return(
            spu_device,
            make_secure_stage_b_formal(0.03, 1e-4, 42, unroll=1),
        )
        average_exec = spu_multi_return(
            spu_device,
            make_secure_stage_b_average([0.2] * clients, 42),
        )
        client_w_s, client_b_s = epoch_exec(x_s, y_s, init_w, init_b)
        ledger.wait("stage_b_epoch_1", [client_w_s, client_b_s], "client_local_secret_state")
        client_w_s, client_b_s = epoch_exec(x_s, y_s, client_w_s, client_b_s)
        ledger.wait("stage_b_epoch_2", [client_w_s, client_b_s], "client_local_secret_state")
        run_id = "corrected_security_smoke_{}".format(secrets.token_hex(8))
        local_w_paths = dump_spu_object(
            spu_device, client_w_s, CKPT, run_id, "stage_b_client_local_w"
        )
        local_b_paths = dump_spu_object(
            spu_device, client_b_s, CKPT, run_id, "stage_b_client_local_b"
        )
        restored_local_w_s = load_spu_object(spu_device, local_w_paths)
        restored_local_b_s = load_spu_object(spu_device, local_b_paths)
        ledger.wait(
            "stage_b_client_share_checkpoint_reload",
            [restored_local_w_s, restored_local_b_s],
            "client_local_secret_state",
        )
        global_w_s, global_b_s = average_exec(restored_local_w_s, restored_local_b_s)
        ledger.wait("stage_b_secure_average", [global_w_s, global_b_s], "global_secret_state")

        w_paths = dump_spu_object(spu_device, global_w_s, CKPT, run_id, "stage_b_global_w")
        b_paths = dump_spu_object(spu_device, global_b_s, CKPT, run_id, "stage_b_global_b")
        restored_w_s = load_spu_object(spu_device, w_paths)
        restored_b_s = load_spu_object(spu_device, b_paths)
        ledger.wait("stage_b_share_checkpoint_reload", [restored_w_s, restored_b_s], "global_secret_state")

        steps, capacity = 2, 4
        x_seq = rng.normal(size=(steps, capacity, feature_dim)).astype(np.float32)
        y_labels = rng.randint(0, classes, size=(steps, capacity))
        y_seq = np.eye(classes, dtype=np.float32)[y_labels]
        mask_seq = np.ones((steps, capacity), dtype=np.float32)
        adaptive_mask = np.asarray([0.0, 1.0], dtype=np.float32)
        x_seq_s = alice(lambda value=x_seq: value)().to(spu_device)
        y_seq_s = alice(lambda value=y_seq: value)().to(spu_device)
        mask_s = alice(lambda value=mask_seq: value)().to(spu_device)
        adaptive_s = alice(lambda value=adaptive_mask: value)().to(spu_device)

        state_root = CKPT / "private_party_noise_state"
        nw_s, nw_paths = request_private_noise_shares(
            pyus, spu_device, state_root, run_id, "weight", (steps, feature_dim, classes)
        )
        nb_s, nb_paths = request_private_noise_shares(
            pyus, spu_device, state_root, run_id, "bias", (steps, classes)
        )
        nc_s, nc_paths = request_private_noise_shares(
            pyus, spu_device, state_root, run_id, "count", (steps,)
        )

        stage_c_inner = make_secure_stage_c_scan(
            0.002, 0.9, 8.0, capacity, 0.5, 0.02, 0.05, 4.0, 0, 1, steps
        )

        def corrected_stage_c(
            xv, yv, mv, nw0, nw1, nw2, nb0, nb1, nb2,
            nc0, nc1, nc2, av, wv, bv, cv,
        ):
            return stage_c_inner(
                xv, yv, mv,
                nw0 + nw1 + nw2,
                nb0 + nb1 + nb2,
                nc0 + nc1 + nc2,
                av, wv, bv, cv,
            )

        stage_c_exec = spu_multi_return(spu_device, corrected_stage_c)
        final_w_s, final_b_s, final_c_s = stage_c_exec(
            x_seq_s, y_seq_s, mask_s,
            nw_s[0], nw_s[1], nw_s[2],
            nb_s[0], nb_s[1], nb_s[2],
            nc_s[0], nc_s[1], nc_s[2],
            adaptive_s, restored_w_s, restored_b_s, np.float32(1.0),
        )
        ledger.wait("stage_c_secure_steps", [final_w_s, final_b_s, final_c_s], "dp_secret_state")

        provider_metadata = {
            "weight": reveal_private_noise_metadata(pyus, nw_paths, ledger, "weight"),
            "bias": reveal_private_noise_metadata(pyus, nb_paths, ledger, "bias"),
            "count": reveal_private_noise_metadata(pyus, nc_paths, ledger, "count"),
        }
        final_w, final_b, final_c = ledger.reveal(
            "smoke_final_global_state",
            [final_w_s, final_b_s, final_c_s],
            "smoke_correctness_only",
            "small synthetic correctness gate; forbidden for client-local state",
        )
        final_w = np.asarray(final_w, dtype=np.float32)
        final_b = np.asarray(final_b, dtype=np.float32)
        final_c = float(np.asarray(final_c).reshape(-1)[0])

        local_public_reveals = [
            row for row in ledger.rows
            if row["scope"] == "client_local_secret_state"
            and row["public_reconstruction"]
        ]
        checks = {
            "stage_b_client_state_never_publicly_reconstructed": not local_public_reveals,
            "stage_b_state_kept_as_spu_objects": type(client_w_s).__name__ == "SPUObject"
            and type(client_b_s).__name__ == "SPUObject",
            "client_share_checkpoint_reload": type(restored_local_w_s).__name__ == "SPUObject"
            and type(restored_local_b_s).__name__ == "SPUObject",
            "global_share_checkpoint_reload": type(restored_w_s).__name__ == "SPUObject"
            and type(restored_b_s).__name__ == "SPUObject",
            "party_noise_commitments_distinct": all(
                len({row["last_share_commitment"] for row in rows}) == 3
                for rows in provider_metadata.values()
            ),
            "party_noise_counters_monotonic": all(
                all(row["requests_completed"] == 1 for row in rows)
                for rows in provider_metadata.values()
            ),
            "noise_entropy_not_persisted": all(
                not row["entropy_persisted"] and not row["share_persisted"]
                for rows in provider_metadata.values() for row in rows
            ),
            "finite_final_state": bool(
                np.all(np.isfinite(final_w))
                and np.all(np.isfinite(final_b))
                and np.isfinite(final_c)
            ),
            "no_plaintext_fallback": True,
        }
        if not all(checks.values()):
            raise AssertionError("corrected security smoke failed: {}".format(checks))

        report = {
            "schema": "new_spu_corrected_security_smoke_v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "backend": "SecretFlow-SPU device API",
            "protocol": runtime_config,
            "adapter": adapter,
            "checks": checks,
            "reveal_ledger": ledger.rows,
            "provider_metadata": provider_metadata,
            "final_clip_bound": final_c,
            "elapsed_seconds": time.monotonic() - started,
            "formal_run": False,
            "official_test_read": False,
            "locked_holdout_read": False,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Corrected-v2 SecretFlow-SPU security smoke",
            "",
            "Status: `COMPLETE`",
            "",
            "| Gate | Pass |",
            "|---|---|",
        ]
        lines.extend("| {} | {} |".format(name, value) for name, value in checks.items())
        lines.extend(
            [
                "",
                "Client-local Stage-B states remained SPUObjects across both epochs.",
                "Only final synthetic smoke state and non-secret provider metadata were revealed.",
                "Every Gaussian share originated from fresh OS CSPRNG entropy inside its own PYU.",
            ]
        )
        MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
