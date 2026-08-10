#!/usr/bin/env python3
"""Synthetic ABY3 smoke for the full-trajectory runner (Phase G0).

Small synthetic data: two Stage-B epochs, one checkpoint evaluation, secret
checkpoint round-trip, and a verification that the training terminal state is
unchanged by evaluation. NOT a formal result.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "new_scripts/new_secretflow_spu"
HERE = Path(__file__).resolve().parent
for location in (str(HERE), str(BASE), str(ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from new_secretflow_spu_backend import register_official_sfl_strategies  # noqa: E402
from new_spu_stage_b_core import (  # noqa: E402
    make_secure_stage_b_average,
    make_secure_stage_b_formal,
)
from new_spu_stage_b_eval_core import make_secure_stage_b_eval_forward  # noqa: E402
from new_spu_corrected_runtime import RevealLedger, spu_multi_return  # noqa: E402

OUT = ROOT / "new_log/new_secretflow_spu/full_trajectory_v1"


def main() -> None:
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    register_official_sfl_strategies()
    names = ["alice", "bob", "carol"]
    sf.init(
        parties=names, address="local", num_cpus=30, omp_num_threads=8,
        log_to_driver=False, include_dashboard=False,
    )
    try:
        pyus = [sf.PYU(name) for name in names]
        alice = pyus[0]
        protocol = {"protocol": "ABY3", "field": "FM128", "fxp_fraction_bits": 16}
        spu = sf.SPU(cluster_def(names, runtime_config=protocol))
        ledger = RevealLedger()
        rng = np.random.RandomState(7)
        dim, classes, clients, per_client, batch_size = 8, 3, 3, 12, 6
        seed = 7
        x_all = rng.rand(clients, per_client, dim).astype(np.float32)
        y_all = rng.randint(0, classes, size=(clients, per_client)).astype(np.int64)
        dev_x = rng.rand(20, dim).astype(np.float32)
        dev_y = rng.randint(0, classes, size=(20,)).astype(np.int64)
        init_w = rng.normal(0, 0.01, (dim, classes)).astype(np.float32)
        init_b = np.zeros(classes, np.float32)
        coeffs = np.ones(clients) / clients
        n_batches = int(np.ceil(per_client / batch_size))
        x_pack = np.zeros((clients, n_batches, batch_size, dim), np.float32)
        y_pack = np.zeros((clients, n_batches, batch_size, classes), np.float32)
        for c in range(clients):
            for bi in range(n_batches):
                sl = slice(bi * batch_size, min((bi + 1) * batch_size, per_client))
                x_pack[c, bi, : len(range(*sl.indices(per_client)))] = x_all[c][sl]
                y_pack[c, bi, : len(range(*sl.indices(per_client)))] = np.eye(classes)[y_all[c][sl]]
        # Use full slices per batch without padding to keep shapes exact.
        x_pack = np.zeros((clients, n_batches, batch_size, dim), np.float32)
        y_pack = np.zeros((clients, n_batches, batch_size, classes), np.float32)
        for c in range(clients):
            idx = np.arange(per_client)
            for bi in range(n_batches):
                sel = idx[bi * batch_size:(bi + 1) * batch_size]
                x_pack[c, bi, :len(sel)] = x_all[c][sel]
                y_pack[c, bi, :len(sel)] = np.eye(classes)[y_all[c][sel]]

        client_w_s = np.stack([init_w for _ in range(clients)])
        client_b_s = np.stack([init_b for _ in range(clients)])
        epoch_exec = spu_multi_return(spu, make_secure_stage_b_formal(0.1, 0.0, seed, 1))
        avg_exec = spu_multi_return(spu, make_secure_stage_b_average(coeffs, seed))
        eval_exec = spu_multi_return(spu, make_secure_stage_b_eval_forward())

        x_s = alice(lambda v=x_pack: v)().to(spu)
        y_s = alice(lambda v=y_pack: v)().to(spu)
        client_w_s, client_b_s = epoch_exec(x_s, y_s, client_w_s, client_b_s)
        ledger.wait("smoke_epoch1", [client_w_s, client_b_s], "client_local_secret_state")
        # One checkpoint evaluation.
        avg_w_s, avg_b_s = avg_exec(client_w_s, client_b_s)
        ledger.wait("smoke_eval_average", [avg_w_s, avg_b_s], "global_secret_state")
        dev_s = alice(lambda v=dev_x: v)().to(spu)
        logits_s = eval_exec(dev_s, avg_w_s, avg_b_s)
        logits = np.asarray(ledger.reveal(
            "smoke_eval_logits", logits_s, "aggregate_development_logits",
            "synthetic smoke evaluation",
        ), dtype=np.float32)
        acc = float(np.mean(np.argmax(logits, axis=1) == dev_y))
        # Second epoch.
        client_w_s, client_b_s = epoch_exec(x_s, y_s, client_w_s, client_b_s)
        ledger.wait("smoke_epoch2", [client_w_s, client_b_s], "client_local_secret_state")

        # Secret checkpoint round-trip: dump and reload, then re-evaluate.
        w_paths = []
        b_paths = []
        ckpt_root = OUT / "synthetic_smoke_ckpt"
        for party in names:
            ckpt_root.mkdir(parents=True, exist_ok=True)
            w_paths.append(str(ckpt_root / "{}_w.pkl".format(party)))
            b_paths.append(str(ckpt_root / "{}_b.pkl".format(party)))
        spu.dump(client_w_s, w_paths)
        spu.dump(client_b_s, b_paths)
        client_w_loaded = spu.load(w_paths)
        client_b_loaded = spu.load(b_paths)
        avg_w2_s, avg_b2_s = avg_exec(client_w_loaded, client_b_loaded)
        logits2_s = eval_exec(dev_s, avg_w2_s, avg_b2_s)
        logits2 = np.asarray(ledger.reveal(
            "smoke_reload_logits", logits2_s, "aggregate_development_logits",
            "synthetic smoke reloaded evaluation",
        ), dtype=np.float32)
        acc2 = float(np.mean(np.argmax(logits2, axis=1) == dev_y))

        report = {
            "schema": "new_spu_full_trajectory_synthetic_smoke_v1",
            "status": "PASS",
            "development_accuracy": acc,
            "reloaded_development_accuracy": acc2,
            "evaluation_matches_after_checkpoint_roundtrip": abs(acc - acc2) < 1e-9,
            "stage_b_client_state_public_reveals": 0,
            "public_reveals": [
                r for r in ledger.rows if r.get("public_reconstruction")
            ],
            "official_test_read": False,
            "locked_holdout_read": False,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "new_spu_full_trajectory_synthetic_smoke.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
