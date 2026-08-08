#!/usr/bin/env python3
"""Run one formal SecretFlow-SPU seed42 experiment with checkpoint resume.

Usage:
    python new_spu_run_formal.py --dataset cifar10 --method true_ours [--resume]

Formal matrix (exactly six, seed 42 only):
    CIFAR-10/100/Tiny-ImageNet x {full_data, true_ours}

Checkpoints live in new_checkpoints/new_secretflow_spu/<run_id>/ (never
committed). Stage-C noise shares are generated from three-party (seed, counter)
state; resume continues the counter so DP noise is never repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

from new_secretflow_spu_backend import register_official_sfl_strategies
from new_spu_pipeline_lib import (
    aggregation_weights,
    deterministic_batches,
    load_ours_cache,
    load_full_cache,
    make_client_splits,
    mtt_anchor_ids,
    one_hot,
    poisson_batch_indices,
    sha256_file,
)
from new_spu_stage_b_core import (
    make_secure_anchor_ce,
    make_secure_stage_b_average,
    make_secure_stage_b_formal,
)
from new_spu_stage_c_core import make_secure_stage_c_scan

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "new_configs/new_secretflow_spu"
OUT_DIR = ROOT / "new_output/new_secretflow_spu"
LOG_DIR = ROOT / "new_log/new_secretflow_spu"
CKPT_DIR = ROOT / "new_checkpoints/new_secretflow_spu"

DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")


def load_config(dataset: str, method: str) -> dict:
    freeze = json.loads(
        (CONFIG_DIR / "new_spu_final_config_freeze.json").read_text(encoding="utf-8")
    )
    return freeze["datasets"][dataset]["methods"][method]


def load_data(dataset: str, method: str, seed: int) -> dict:
    """Load training features + reference/dev splits for the given run."""
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny": 200}[dataset]
    if method == "full_data":
        b_cache_path = (
            "new_cache/new_true_ours/full_data_stage_b/{}/new_full_stage_b_{}_seed{}.npz".format(
                dataset, dataset, seed
            )
        )
        c_cache_path = (
            "new_cache/new_true_ours/full_data_stage_c/{}/new_full_stage_c_{}_seed{}.npz".format(
                dataset, dataset, seed
            )
        )
        b_cache = load_full_cache(b_cache_path)
        c_cache = load_full_cache(c_cache_path)
        cache_mode = "memmap"
    else:
        b_cache_path = (
            "new_cache/new_true_ours/official_mtt/stage_b/{}/new_true_ours_stage_b_{}_seed{}.npz".format(
                dataset, dataset, seed
            )
        )
        c_cache_path = (
            "new_cache/new_true_ours/official_mtt/stage_c/{}/new_true_ours_stage_c_{}_seed{}.npz".format(
                dataset, dataset, seed
            )
        )
        b_cache = load_ours_cache(b_cache_path)
        c_cache = load_ours_cache(c_cache_path)
        cache_mode = "npz"
    return {
        "num_classes": num_classes,
        "b_x": b_cache["x_encoder_train"],
        "b_y": b_cache["y_encoder_train"],
        "b_x_ref": b_cache["x_reference"],
        "b_y_ref": b_cache["y_reference"],
        "c_x": c_cache["x_encoder_train"],
        "c_y": c_cache["y_encoder_train"],
        "c_x_dev": c_cache["x_reference"],
        "c_y_dev": c_cache["y_reference"],
        "b_cache_path": b_cache_path,
        "c_cache_path": c_cache_path,
        "cache_mode": cache_mode,
    }


_STAGE_B_POOL_CACHE = {}


def _load_stage_b_pool(cache_path, cache_mode):
    """Return (train_x float32, train_y int64) for the Stage-B pool.

    Caches the loaded pool inside the PYU worker process (module-level dict),
    so repeated epoch calls never re-read the memmap or npz from disk.
    """
    key = "{}:{}".format(cache_path, cache_mode)
    if key in _STAGE_B_POOL_CACHE:
        return _STAGE_B_POOL_CACHE[key]
    if cache_mode == "npz":
        with np.load(cache_path, allow_pickle=False) as z:
            train_x = np.asarray(z["x_encoder_train"], dtype=np.float32)
            train_y = np.asarray(z["y_encoder_train"], dtype=np.int64)
    else:
        # memmap-backed Full path: use the frozen train indices from metadata.
        meta = np.load(cache_path, allow_pickle=False)
        feature_npy_path = str(
            np.asarray(meta["feature_npy_path"]).reshape(-1)[0], encoding="utf-8"
        )
        train_idx = np.asarray(meta["train_indices"], dtype=np.int64)
        mmap = np.load(feature_npy_path, mmap_mode="r")
        train_x = np.asarray(mmap[train_idx], dtype=np.float32)
        train_y = np.asarray(meta["y_encoder_train"], dtype=np.int64)
    _STAGE_B_POOL_CACHE[key] = (train_x, train_y)
    return train_x, train_y


def build_x_pack(
    cache_path,
    cache_mode,
    client_indices,
    epoch,
    num_clients,
    batch_size,
    feature_dim,
    seed,
):
    """PYU-side feature batch packer (single array; captures path+indices only,
    never a large feature array). Returns x_pack float32 (C,B,M,D)."""
    train_x, _ = _load_stage_b_pool(cache_path, cache_mode)
    per_client = len(client_indices[0])
    batch_count = int(np.ceil(per_client / float(batch_size)))
    x_pack = np.zeros(
        (num_clients, batch_count, batch_size, feature_dim), dtype=np.float32
    )
    for cid in range(num_clients):
        batches = deterministic_batches(
            client_indices[cid], batch_size, seed, 1, epoch, cid
        )
        for bi, bidx in enumerate(batches):
            x_pack[cid, bi] = train_x[bidx]
    return x_pack


def build_y_pack(
    cache_path,
    cache_mode,
    client_indices,
    epoch,
    num_clients,
    batch_size,
    num_classes,
    seed,
):
    """PYU-side label packer (single array). Returns y_pack_hot (C,B,M,K)."""
    _, train_y = _load_stage_b_pool(cache_path, cache_mode)
    per_client = len(client_indices[0])
    batch_count = int(np.ceil(per_client / float(batch_size)))
    y_pack = np.zeros((num_clients, batch_count, batch_size, 1), dtype=np.int64)
    for cid in range(num_clients):
        batches = deterministic_batches(
            client_indices[cid], batch_size, seed, 1, epoch, cid
        )
        for bi, bidx in enumerate(batches):
            y_pack[cid, bi, :, 0] = train_y[bidx]
    y_pack_hot = one_hot(y_pack.reshape(-1), num_classes).reshape(
        num_clients, batch_count, batch_size, num_classes
    )
    return y_pack_hot


def stage_b_initializer(
    spu_device,
    data,
    config,
    run_id,
    dataset,
    seed,
    num_classes,
    cache_path,
    cache_mode,
    log,
):
    """Run formal Stage-B (epoch loop in Python, mini-batches in SPU), save the
    refined initializer, return (weight, bias, stage_b_seconds)."""
    feature_dim = data["b_x"].shape[1]
    b_cfg = config["stage_b"]
    # Fixed formal protocol constants (TFE freeze): 5 clients, balanced split.
    num_clients = int(b_cfg.get("num_clients", 5))
    split_kind = b_cfg.get("split_kind", "balanced")
    # Ours uses the labeled compact-anchor refinement; Full does not.
    if config["anchor_ce"]:
        local_ids, anchor_ids = mtt_anchor_ids(data["b_y"], num_classes)
        x_local, y_local = data["b_x"][local_ids], data["b_y"][local_ids]
        x_anchor, y_anchor = data["b_x"][anchor_ids], data["b_y"][anchor_ids]
        client_indices = make_client_splits(
            y_local, num_clients, split_kind, seed
        )
        train_y = y_local
    else:
        x_anchor = np.zeros((0, feature_dim), dtype=np.float32)
        y_anchor = np.zeros((0,), dtype=np.int64)
        client_indices = make_client_splits(
            data["b_y"], num_clients, split_kind, seed
        )
        train_y = data["b_y"]
    coeffs = aggregation_weights(train_y, client_indices, b_cfg["client_weighting"])
    batch_size = int(b_cfg["batch_size"])
    if "batch_override" in config["stage_b"] and config["stage_b"]["batch_override"]:
        batch_size = int(config["stage_b"]["batch_override"])
    local_epochs = int(b_cfg["local_epochs"])
    fedavg_sync_interval = int(b_cfg["fedavg_sync_interval"])

    rng = np.random.RandomState(seed)
    init_w = rng.normal(0.0, 0.01, size=(feature_dim, num_classes)).astype(np.float32)
    init_b = np.zeros((num_classes,), dtype=np.float32)
    client_w = np.stack([init_w for _ in range(num_clients)], axis=0)
    client_b = np.stack([init_b for _ in range(num_clients)], axis=0)

    alice = sf.PYU("alice")
    # Choose a scan unroll factor that divides num_batches (SPU vectorization);
    # fall back to 1 if num_batches is prime or no suitable factor exists.
    num_batches_total = int(np.ceil(len(client_indices[0]) / float(batch_size)))
    if "unroll_override" in b_cfg and b_cfg["unroll_override"]:
        unroll = int(b_cfg["unroll_override"])
    else:
        unroll = 1
        for candidate in (64, 32, 23, 19, 16, 11, 10, 8, 5, 4, 2):
            if num_batches_total % candidate == 0:
                unroll = candidate
                break
    print(
        "[StageB] num_batches={} unroll={}".format(num_batches_total, unroll),
        flush=True,
    )
    epoch_fn = spu_device(
        make_secure_stage_b_formal(
            b_cfg["local_lr"], b_cfg["weight_decay"], seed, unroll=unroll
        )
    )
    avg_fn = spu_device(
        make_secure_stage_b_average(coeffs, seed)
    )
    anchor_fn = (
        spu_device(
            make_secure_anchor_ce(
                config["anchor_steps"], config["anchor_lr"], config["anchor_mu"], seed
            )
        )
        if config["anchor_ce"]
        else None
    )
    pack_x_fn = alice(build_x_pack)
    pack_y_fn = alice(build_y_pack)

    stage_b_started = time.monotonic()
    for epoch in range(local_epochs):
        epoch_started = time.monotonic()
        # Build this epoch's packs inside the PYU worker (captures path+indices
        # only, never a large feature array), then secret-share directly into
        # SPU without materializing the array on the driver host.
        x_s = pack_x_fn(
            cache_path, cache_mode, client_indices, epoch,
            num_clients, batch_size, feature_dim, seed,
        ).to(spu_device)
        y_s = pack_y_fn(
            cache_path, cache_mode, client_indices, epoch,
            num_clients, batch_size, num_classes, seed,
        ).to(spu_device)
        client_w, client_b = sf.reveal(
            epoch_fn(x_s, y_s, client_w, client_b)
        )
        client_w = np.asarray(client_w, dtype=np.float32)
        client_b = np.asarray(client_b, dtype=np.float32)
        if fedavg_sync_interval > 0 and (epoch + 1) % fedavg_sync_interval == 0:
            avg_w, avg_b = sf.reveal(avg_fn(client_w, client_b))
            client_w = np.stack([np.asarray(avg_w, dtype=np.float32) for _ in range(num_clients)], axis=0)
            client_b = np.stack([np.asarray(avg_b, dtype=np.float32) for _ in range(num_clients)], axis=0)
        if (epoch + 1) % 2 == 0:
            print(
                "[StageB] epoch={}/{} epoch_wall={:.1f}s".format(
                    epoch + 1, local_epochs, time.monotonic() - epoch_started
                ),
                flush=True,
            )

    avg_w, avg_b = sf.reveal(avg_fn(client_w, client_b))
    w = np.asarray(avg_w, dtype=np.float32)
    b = np.asarray(avg_b, dtype=np.float32)
    if config["anchor_ce"]:
        anchor_x_s = alice(lambda v=x_anchor.astype(np.float32): v)().to(spu_device)
        anchor_y_s = alice(lambda v=one_hot(y_anchor, num_classes): v)().to(spu_device)
        w, b = sf.reveal(anchor_fn(anchor_x_s, anchor_y_s, w, b))
        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
    stage_b_seconds = time.monotonic() - stage_b_started

    ckpt_dir = CKPT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ckpt_dir / "stage_b_initializer.npz",
        weight=w,
        bias=b,
        anchor_ce=config["anchor_ce"],
        dataset=dataset,
        method=config["anchor_ce"] and "true_ours" or "full_data",
        seed=seed,
        stage_b_seconds=stage_b_seconds,
    )
    return w, b, stage_b_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--block_steps", type=int, default=10)
    parser.add_argument("--batch_override", type=int, default=None)
    parser.add_argument("--unroll_override", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    config = load_config(args.dataset, args.method)
    if args.batch_override:
        config["stage_b"]["batch_override"] = args.batch_override
    if args.unroll_override:
        config["stage_b"]["unroll_override"] = args.unroll_override
    data = load_data(args.dataset, args.method, args.seed)
    num_classes = data["num_classes"]
    run_id = "spu_{}_{}_seed{}".format(args.dataset, args.method, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = OUT_DIR / "{}_{}_seed{}_trace.csv".format(
        args.dataset, args.method, args.seed
    )
    report_path = LOG_DIR / "new_spu_{}_{}_seed{}_report.json".format(
        args.dataset, args.method, args.seed
    )

    adapter = register_official_sfl_strategies()
    parties = ["alice", "bob", "carol"]
    # Ensure Ray PYU workers inherit the repo + spu-scripts module paths.
    spu_scripts = str(ROOT / "new_scripts/new_secretflow_spu")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = spu_scripts + ":" + str(ROOT) + (":" + existing if existing else "")
    sf.init(
        parties=parties,
        address="local",
        num_cpus=30,
        omp_num_threads=8,
        log_to_driver=False,
        include_dashboard=False,
    )
    try:
        runtime_config = {"protocol": "ABY3", "field": "FM128", "fxp_fraction_bits": 16}
        cdef = cluster_def(parties, runtime_config=runtime_config)
        spu_device = sf.SPU(cdef)

        report = {
            "schema": "new_spu_formal_run_v1",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "run_id": run_id,
            "backend": "SecretFlow-SPU device API",
            "protocol": runtime_config,
            "status": "RUNNING",
            "locked_holdout_read": False,
            "official_test_read": False,
        }

        # Stage-B
        w, b, stage_b_seconds = stage_b_initializer(
            spu_device, data, config, run_id, args.dataset, args.seed,
            num_classes, data["b_cache_path"], data["cache_mode"], None,
        )
        report["stage_b_seconds"] = stage_b_seconds

        # Stage-C formal
        sc = config["stage_c"]
        capacity = int(sc["graph_capacity"])
        expected_batch = int(sc["expected_batch"])
        block_steps = int(args.block_steps)
        total_steps = int(sc["steps"])

        ckpt_dir = CKPT_DIR / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "stage_c_ckpt.json"
        noise_path = ckpt_dir / "stage_c_noise_state.npz"
        if args.resume and ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            w = np.asarray(ckpt["weight"], dtype=np.float32)
            b = np.asarray(ckpt["bias"], dtype=np.float32)
            c = float(ckpt["clip_bound"])
            steps_done = int(ckpt["steps_done"])
            with np.load(noise_path, allow_pickle=False) as nz:
                party_seeds = [int(v) for v in nz["party_seeds"]]
                counters = [int(v) for v in nz["counters"]]
            print("[resume] steps_done={}".format(steps_done), flush=True)
        else:
            c = float(sc["clip_c"])
            steps_done = 0
            party_seeds = [int(x) for x in np.random.RandomState(seed=args.seed).randint(0, 2**31, size=3)]
            counters = [0, 0, 0]

        # Pre-generate adaptive mask over the whole run (static).
        adaptive_mask = np.asarray(
            [
                1.0
                if step >= int(sc["warmup_steps"])
                and (step + 1) % int(sc["update_interval"]) == 0
                else 0.0
                for step in range(total_steps)
            ],
            dtype=np.float32,
        )

        stage_c_online_started = time.monotonic()
        trace_rows = []
        while steps_done < total_steps:
            block = min(block_steps, total_steps - steps_done)
            # Three-party noise shares for this block from (seed, counter).
            block_w = []
            block_b = []
            block_n = []
            for p in range(3):
                base = (int(party_seeds[p]) * 1000003 + int(counters[p])) % (2**32)
                rng = np.random.RandomState(seed=base)
                block_w.append(
                    rng.normal(0.0, 1.0 / np.sqrt(3.0), size=(block, data["c_x"].shape[1], num_classes)).astype(np.float32)
                )
                rng = np.random.RandomState(seed=(base + 1) % (2**32))
                block_b.append(rng.normal(0.0, 1.0 / np.sqrt(3.0), size=(block, num_classes)).astype(np.float32))
                rng = np.random.RandomState(seed=(base + 2) % (2**32))
                block_n.append(rng.normal(0.0, 1.0 / np.sqrt(3.0), size=(block,)).astype(np.float32))

            # Assemble block batch sequences.
            block_ids = []
            for s in range(block):
                step = steps_done + s
                ids = poisson_batch_indices(len(data["c_x"]), expected_batch, random_bytes=os.urandom)
                block_ids.append(ids)
            x_seq = np.zeros((block, capacity, data["c_x"].shape[1]), dtype=np.float32)
            y_seq = np.zeros((block, capacity, num_classes), dtype=np.float32)
            m_seq = np.zeros((block, capacity), dtype=np.float32)
            for s, ids in enumerate(block_ids):
                n = len(ids)
                x_seq[s, :n] = np.asarray(data["c_x"][ids], dtype=np.float32)
                y_seq[s, :n] = one_hot(np.asarray(data["c_y"][ids], dtype=np.int64), num_classes)
                m_seq[s, :n] = 1.0
            am_block = adaptive_mask[steps_done : steps_done + block]

            alice = sf.PYU("alice")
            bob = sf.PYU("bob")
            carol = sf.PYU("carol")
            x_s = alice(lambda v=x_seq: v)().to(spu_device)
            y_s = alice(lambda v=y_seq: v)().to(spu_device)
            m_s = alice(lambda v=m_seq: v)().to(spu_device)
            am_s = alice(lambda v=am_block: v)().to(spu_device)
            nw_s = [
                alice(lambda v=block_w[p]: v)().to(spu_device) for p in range(3)
            ]
            nb_s = [
                bob(lambda v=block_b[p]: v)().to(spu_device) for p in range(3)
            ]
            nn_s = [
                carol(lambda v=block_n[p]: v)().to(spu_device) for p in range(3)
            ]

            stage_c_fn = make_secure_stage_c_scan(
                sc["learning_rate"], sc["noise_multiplier"],
                sc["count_noise_multiplier"], expected_batch,
                sc["gamma"], sc["eta_c"], sc["c_min"], sc["c_max"],
                sc["warmup_steps"], sc["update_interval"], block,
            )

            def make_run():
                def _run(xv, yv, mv, nw0, nw1, nw2, nb0, nb1, nb2, nn0, nn1, nn2, amv, wv, bv, cv):
                    return stage_c_fn(
                        xv, yv, mv,
                        nw0 + nw1 + nw2, nb0 + nb1 + nb2, nn0 + nn1 + nn2,
                        amv, wv, bv, cv,
                    )

                _run.__name__ = "stage_c_block_run"
                return _run

            block_started = time.monotonic()
            nw, nb, nc = sf.reveal(spu_device(make_run())(
                x_s, y_s, m_s,
                nw_s[0], nw_s[1], nw_s[2],
                nb_s[0], nb_s[1], nb_s[2],
                nn_s[0], nn_s[1], nn_s[2],
                am_s, w, b, c,
            ))
            w = np.asarray(nw, dtype=np.float32)
            b = np.asarray(nb, dtype=np.float32)
            c = float(np.asarray(nc).reshape(-1)[0])
            block_seconds = time.monotonic() - block_started
            steps_done += block
            counters = [c + block for c in counters]

            # Development evaluation at block boundary.
            dev_logits = data["c_x_dev"] @ w + b
            dev_acc = float(
                np.mean(
                    np.argmax(dev_logits, axis=1) == np.asarray(data["c_y_dev"], dtype=np.int64)
                )
            )
            cumulative_online = time.monotonic() - stage_c_online_started
            trace_rows.append(
                {
                    "block": steps_done // block,
                    "steps_done": steps_done,
                    "stage_c_online_secure_seconds": round(cumulative_online, 4),
                    "block_seconds": round(block_seconds, 4),
                    "clip_bound": round(c, 6),
                    "development_accuracy": round(dev_acc, 6),
                }
            )

            # Save checkpoint (weight/bias/clip/steps/counters) - no noise repeat.
            np.savez(
                noise_path,
                party_seeds=np.asarray(party_seeds, dtype=np.int64),
                counters=np.asarray(counters, dtype=np.int64),
            )
            ckpt = {
                "weight": w.tolist(),
                "bias": b.tolist(),
                "clip_bound": c,
                "steps_done": steps_done,
            }
            ckpt_path.write_text(json.dumps(ckpt), encoding="utf-8")
            print(
                "[StageC] steps={}/{} dev_acc={:.4f} c={:.5f} online={:.1f}s".format(
                    steps_done, total_steps, dev_acc, c, cumulative_online
                ),
                flush=True,
            )

        # Trace CSV.
        import csv

        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)

        report["status"] = "COMPLETE"
        report["final_development_accuracy"] = trace_rows[-1]["development_accuracy"]
        report["final_clip_bound"] = c
        report["cumulative_stage_c_online_secure_seconds"] = trace_rows[-1][
            "stage_c_online_secure_seconds"
        ]
        report["total_online_secure_seconds"] = (
            stage_b_seconds + report["cumulative_stage_c_online_secure_seconds"]
        )
        report["peak_driver_rss_gib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1048576.0
        report["checkpoint"] = str(ckpt_dir)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
