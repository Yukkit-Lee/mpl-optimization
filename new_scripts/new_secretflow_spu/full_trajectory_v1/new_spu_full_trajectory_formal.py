#!/usr/bin/env python3
"""Full SPU end-to-end trajectory runner (seed42), schema new_spu_full_trajectory_v1.

Rebuilds the six corrected-v2 cells from Stage-B epoch 0 with preregistered
checkpoint evaluations, so Figure 2 can show a continuous real Stage-B ->
Stage-C trajectory.

Stage-B stays fully secret-shared:
- five client heads remain SPUObjects across epochs;
- FedAvg/average happen inside SPU;
- at preregistered checkpoint epochs, a *temporary* weighted average of the
  secret client heads is computed inside SPU, development features are pushed
  through it in chunks inside SPU, and ONLY the aggregate development logits
  are revealed;
- no averaged weight/bias, no client head, no per-client logits are revealed;
- development labels are used on the driver only to compute aggregate accuracy;
- evaluation is side-effect free and never enters the training-online timer.

Stage C reuses the audited corrected-v2 adaptive-DP semantics with three-party
private OS-CSPRNG noise shares and the Opacus RDP accountant.

Two time fields per trace point:
  cumulative_training_online_secure_seconds  (main T@Target axis; no eval overhead)
  cumulative_instrumented_wall_seconds       (compile/infeed/checkpoint/eval included)
"""

from __future__ import annotations

import argparse
import csv
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
from new_spu_pipeline_lib import (  # noqa: E402
    aggregation_weights,
    deterministic_batches,
    make_client_splits,
    mtt_anchor_ids,
    one_hot,
    poisson_batch_indices,
)
from new_spu_run_formal import load_config, load_data, _load_stage_b_pool  # noqa: E402
from new_spu_stage_b_core import (  # noqa: E402
    make_secure_anchor_ce,
    make_secure_stage_b_average,
    make_secure_stage_b_average_and_broadcast,
    make_secure_stage_b_formal,
)
from new_spu_stage_c_core import make_secure_stage_c_scan  # noqa: E402
from new_spu_corrected_audit import (  # noqa: E402
    CgroupMemoryMonitor,
    PhaseTimer,
    cgroup_delta,
    cgroup_snapshot,
    conditional_stage_c_epsilon,
)
from new_spu_corrected_runtime import (  # noqa: E402
    RevealLedger,
    dump_spu_object,
    load_spu_object,
    request_private_noise_shares,
    reveal_private_noise_metadata,
    spu_multi_return,
)
from new_spu_stage_b_eval_core import (  # noqa: E402
    assert_chunk_cover,
    make_secure_stage_b_eval_forward,
    split_development_rows,
)

OUT = ROOT / "new_output/new_secretflow_spu/full_trajectory_v1"
LOG = ROOT / "new_log/new_secretflow_spu/full_trajectory_v1"
CKPT = ROOT / "new_checkpoints/new_secretflow_spu/full_trajectory_v1"
DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")
SCHEMA = "new_spu_full_trajectory_v1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_trace(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def merge_prior_timing(timer: PhaseTimer, prior: dict) -> None:
    current = dict(timer.seconds)
    timer.seconds = {str(key): float(value) for key, value in prior.items()}
    for phase, seconds in current.items():
        timer.add(phase, seconds)


def build_private_epoch_packs(
    cache_path, cache_mode, source_indices, client_indices, epoch,
    batch_size, feature_dim, num_classes, seed, batch_start, batch_count,
):
    """PYU-only Stage-B packer (reused corrected-v2 packer)."""
    train_x, train_y = _load_stage_b_pool(cache_path, cache_mode)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    clients = len(client_indices)
    x_pack = np.zeros((clients, batch_count, batch_size, feature_dim), dtype=np.float32)
    y_pack = np.zeros((clients, batch_count, batch_size, num_classes), dtype=np.float32)
    for client_id in range(clients):
        all_batches = deterministic_batches(
            client_indices[client_id], batch_size, seed, 1, epoch, client_id
        )
        selected_batches = all_batches[batch_start : batch_start + batch_count]
        if len(selected_batches) != batch_count:
            raise ValueError("requested Stage-B batch chunk exceeds epoch")
        for batch_id, local_ids in enumerate(selected_batches):
            original_ids = source_indices[np.asarray(local_ids, dtype=np.int64)]
            x_pack[client_id, batch_id] = train_x[original_ids]
            y_pack[client_id, batch_id] = one_hot(train_y[original_ids], num_classes)
    return x_pack, y_pack


def corrected_stage_c_function(stage_c, expected_batch, block):
    inner = make_secure_stage_c_scan(
        stage_c["learning_rate"], stage_c["noise_multiplier"],
        stage_c["count_noise_multiplier"], expected_batch,
        stage_c["gamma"], stage_c["eta_c"], stage_c["c_min"], stage_c["c_max"],
        stage_c["warmup_steps"], stage_c["update_interval"], block,
    )

    def _run(
        xv, yv, mv, nw0, nw1, nw2, nb0, nb1, nb2,
        nc0, nc1, nc2, av, wv, bv, cv,
    ):
        return inner(
            xv, yv, mv, nw0 + nw1 + nw2, nb0 + nb1 + nb2,
            nc0 + nc1 + nc2, av, wv, bv, cv,
        )

    _run.__name__ = "corrected_stage_c_block_{}".format(block)
    return _run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--batch_chunk_batches", type=int, required=True)
    parser.add_argument("--local_lr", type=float)
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--block_steps", type=int, default=10)
    parser.add_argument("--stage_b_eval_interval", type=int, default=5)
    parser.add_argument("--stage_b_eval_chunk_rows", type=int, default=250)
    parser.add_argument("--no_stage_b_eval", action="store_true")
    parser.add_argument("--max_stage_b_epochs", type=int)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--stage_b_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    config = load_config(args.dataset, args.method)
    data = load_data(args.dataset, args.method, args.seed)
    num_classes = int(data["num_classes"])
    feature_dim = int(data["b_x"].shape[1])
    run_id = "full_trajectory_v1_formal_{}_{}_seed42{}".format(
        args.dataset, args.method, "_" + args.run_tag if args.run_tag else ""
    )
    output_dir = OUT / run_id
    checkpoint_dir = CKPT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "new_spu_full_trajectory_trace.csv"
    report_path = LOG / "{}_report.json".format(run_id)
    LOG.mkdir(parents=True, exist_ok=True)
    if args.resume and report_path.exists():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        if completed.get("status") == "COMPLETE":
            print(json.dumps(completed, sort_keys=True), flush=True)
            return

    bcfg = config["stage_b"]
    stage_c = config["stage_c"]
    eval_interval = int(args.stage_b_eval_interval)
    eval_chunk_rows = int(args.stage_b_eval_chunk_rows)
    effective = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "stage_b_batch_size": args.batch_size,
        "stage_b_batch_chunk_batches": args.batch_chunk_batches,
        "stage_b_batch_size_source": "explicit_pre_run_freeze",
        "stage_b_unroll": args.unroll,
        "stage_b_local_lr": float(
            args.local_lr if args.local_lr is not None else bcfg["local_lr"]
        ),
        "stage_b_shared_initialization": bool(bcfg["shared_initialization"]),
        "stage_b_local_epochs": int(
            min(int(bcfg["local_epochs"]), args.max_stage_b_epochs or int(bcfg["local_epochs"]))
        ),
        "stage_c_steps": int(stage_c["steps"]),
        "stage_c_block_steps": args.block_steps,
        "stage_b_eval_interval_epochs": eval_interval,
        "stage_b_eval_chunk_rows": eval_chunk_rows,
        "stage_b_eval_enabled": not args.no_stage_b_eval,
        "run_kind": "development_gate" if args.max_stage_b_epochs else "formal",
        "cpu": {
            "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
            "num_cpus": 30,
            "taskset_cpuset": os.environ.get("TASKSSET_CPUSET", "unknown"),
        },
        "formal_run": not args.max_stage_b_epochs,
    }
    atomic_json(output_dir / "effective_config.json", {"base": config, "effective": effective})

    adapter = register_official_sfl_strategies()
    names = ["alice", "bob", "carol"]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join([str(HERE), str(BASE), str(ROOT), existing])
    sf.init(
        parties=names, address="local", num_cpus=30, omp_num_threads=8,
        log_to_driver=False, include_dashboard=False,
    )
    timer = PhaseTimer()
    ledger = RevealLedger()
    resource_start = cgroup_snapshot()
    memory_monitor = CgroupMemoryMonitor()
    memory_monitor.start()
    started = time.monotonic()
    elapsed_offset = 0.0
    try:
        pyus = [sf.PYU(name) for name in names]
        alice = pyus[0]
        protocol = {"protocol": "ABY3", "field": "FM128", "fxp_fraction_bits": 16}
        spu = sf.SPU(cluster_def(names, runtime_config=protocol))
        report = {
            "schema": SCHEMA,
            "status": "RUNNING",
            "run_id": run_id,
            "run_kind": "formal",
            "backend": "SecretFlow-SPU device API",
            "protocol": protocol,
            "adapter": adapter,
            "effective_config": effective,
            "confidentiality_scope": (
                "Stage-B client-local and global state remain secret-shared; "
                "only preregistered aggregate development logits are public"
            ),
            "privacy_scope": "conditional Stage-C record-level DP only; Stage-B is non-DP",
            "official_test_read": False,
            "locked_holdout_read": False,
        }

        # ----- Stage B -----
        if config["anchor_ce"]:
            local_source_ids, anchor_ids = mtt_anchor_ids(data["b_y"], num_classes)
            split_labels = np.asarray(data["b_y"])[local_source_ids]
            anchor_x = np.asarray(data["b_x"])[anchor_ids]
            anchor_y = np.asarray(data["b_y"])[anchor_ids]
        else:
            local_source_ids = np.arange(len(data["b_y"]), dtype=np.int64)
            split_labels = np.asarray(data["b_y"])
            anchor_x = np.zeros((0, feature_dim), dtype=np.float32)
            anchor_y = np.zeros((0,), dtype=np.int64)
        client_indices = make_client_splits(
            split_labels, 5, bcfg.get("split_kind", "balanced"), args.seed
        )
        coefficients = aggregation_weights(
            split_labels, client_indices, bcfg["client_weighting"]
        )
        batches = int(np.ceil(len(client_indices[0]) / float(args.batch_size)))
        chunk_batches = int(args.batch_chunk_batches)
        if chunk_batches <= 0 or chunk_batches > batches:
            raise ValueError("batch_chunk_batches must be in [1, per-client batches]")
        if chunk_batches % args.unroll:
            raise ValueError("unroll must divide batch_chunk_batches")
        remainder_batches = batches % chunk_batches
        if remainder_batches and remainder_batches % args.unroll:
            raise ValueError("unroll must also divide the final batch chunk")

        rng = np.random.RandomState(args.seed)
        initial_weights = [
            rng.normal(0.0, 0.01, size=(feature_dim, num_classes)).astype(np.float32)
            for _ in range(5)
        ]
        if bool(bcfg["shared_initialization"]):
            initial_weights = [initial_weights[0].copy() for _ in range(5)]
        initial_biases = [np.zeros((num_classes,), dtype=np.float32) for _ in range(5)]
        client_w_s = np.stack(initial_weights, axis=0)
        client_b_s = np.stack(initial_biases, axis=0)
        epoch_exec = spu_multi_return(
            spu, make_secure_stage_b_formal(
                effective["stage_b_local_lr"], bcfg["weight_decay"], args.seed, args.unroll
            )
        )
        average_exec = spu_multi_return(
            spu, make_secure_stage_b_average(coefficients, args.seed)
        )
        sync_exec = spu_multi_return(
            spu, make_secure_stage_b_average_and_broadcast(coefficients, args.seed)
        )
        eval_forward_exec = None
        if not args.no_stage_b_eval:
            eval_forward_exec = spu_multi_return(
                spu, make_secure_stage_b_eval_forward()
            )
        pack_exec = alice(build_private_epoch_packs)

        # Compile Stage-B epoch graph with shape-identical private packs.
        with timer.measure("stage_b_compile_warmup"):
            warm = pack_exec(
                data["b_cache_path"], data["cache_mode"], local_source_ids,
                client_indices, 0, args.batch_size, feature_dim, num_classes, args.seed,
                0, chunk_batches,
            )
            warm_x_s = alice(lambda pair: pair[0])(warm).to(spu)
            warm_y_s = alice(lambda pair: pair[1])(warm).to(spu)
            warm_w_s, warm_b_s = epoch_exec(warm_x_s, warm_y_s, client_w_s, client_b_s)
            sf.wait([warm_w_s, warm_b_s])
            if remainder_batches:
                warm_tail = pack_exec(
                    data["b_cache_path"], data["cache_mode"], local_source_ids,
                    client_indices, 0, args.batch_size, feature_dim, num_classes,
                    args.seed, batches - remainder_batches, remainder_batches,
                )
                warm_tail_x_s = alice(lambda pair: pair[0])(warm_tail).to(spu)
                warm_tail_y_s = alice(lambda pair: pair[1])(warm_tail).to(spu)
                tail_w_s, tail_b_s = epoch_exec(
                    warm_tail_x_s, warm_tail_y_s, client_w_s, client_b_s
                )
                sf.wait([tail_w_s, tail_b_s])

        # Compile the Stage-B evaluation graph (chunked forward) with public shapes.
        dev_rows = int(data["c_x_dev"].shape[0])
        chunks = split_development_rows(dev_rows, eval_chunk_rows)
        assert_chunk_cover(chunks, dev_rows)
        if not args.no_stage_b_eval:
            with timer.measure("stage_b_eval_compile_warmup"):
                z_dev = np.zeros((eval_chunk_rows, feature_dim), np.float32)
                z_w = np.zeros((feature_dim, num_classes), np.float32)
                z_b = np.zeros((num_classes,), np.float32)
                warm_logits = eval_forward_exec(z_dev, z_w, z_b)
                sf.wait([warm_logits])

        stage_b_meta = checkpoint_dir / "stage_b_progress.json"
        stage_b_trace = output_dir / "stage_b_trace.csv"
        state_meta = checkpoint_dir / "stage_c_progress.json"
        start_epoch = 0
        stage_b_complete = False
        trace_rows = []
        if args.resume and stage_b_meta.exists():
            state = json.loads(stage_b_meta.read_text(encoding="utf-8"))
            if not state_meta.exists() and state.get("timing_seconds"):
                merge_prior_timing(timer, state["timing_seconds"])
                elapsed_offset = float(state.get("raw_wall_seconds", 0.0))
            start_epoch = int(state["epochs_done"])
            stage_b_complete = bool(state.get("stage_b_complete", False))
            if stage_b_complete:
                global_w_s = load_spu_object(spu, state["global_w_share_paths"])
                global_b_s = load_spu_object(spu, state["global_b_share_paths"])
                ledger.wait(
                    "stage_b_complete_resume", [global_w_s, global_b_s],
                    "global_secret_state",
                )
            else:
                client_w_s = load_spu_object(spu, state["client_w_share_paths"])
                client_b_s = load_spu_object(spu, state["client_b_share_paths"])
                ledger.wait(
                    "stage_b_resume", [client_w_s, client_b_s],
                    "client_local_secret_state",
                )
            if stage_b_trace.exists():
                with stage_b_trace.open(newline="", encoding="utf-8") as handle:
                    trace_rows = list(csv.DictReader(handle))

        def stage_b_evaluate(epoch_or_step, stage_label, ledger, timer, spu):
            """Side-effect-free aggregate dev evaluation of the 5 secret heads."""
            # Temporary weighted average inside SPU; do NOT touch client_w_s/client_b_s.
            with timer.measure("stage_b_evaluation_online_secure"):
                avg_w_s, avg_b_s = average_exec(client_w_s, client_b_s)
                ledger.wait(
                    "stage_b_eval_{}_temp_average".format(stage_label),
                    [avg_w_s, avg_b_s], "global_secret_state",
                )
                logits_chunks = []
                for idx, (start, end) in enumerate(chunks):
                    rows_now = end - start
                    dev_chunk = np.asarray(data["c_x_dev"][start:end], dtype=np.float32)
                    logits_s = eval_forward_exec(dev_chunk, avg_w_s, avg_b_s)
                    logits_chunks.append(logits_s)
                sf.wait(logits_chunks)
                logits_flat = []
                for idx, logits_s in enumerate(logits_chunks):
                    revealed = ledger.reveal(
                        "stage_b_eval_{}_chunk_{}".format(stage_label, idx),
                        logits_s, "aggregate_development_logits",
                        "preregistered Stage-B checkpoint aggregate dev logits",
                    )
                    logits_flat.append(np.asarray(revealed, dtype=np.float32))
            logits = np.concatenate(logits_flat, axis=0)
            if logits.shape[0] != dev_rows or logits.shape[1] != num_classes:
                raise AssertionError(
                    "development logits shape mismatch: {} vs expected ({},{})".format(
                        logits.shape, dev_rows, num_classes
                    )
                )
            labels = np.asarray(data["c_y_dev"], dtype=np.int64)
            acc = float(np.mean(np.argmax(logits, axis=1) == labels))
            train_online = float(
                timer.seconds.get("stage_b_online_secure", 0.0)
                + timer.seconds.get("stage_c_online_secure", 0.0)
            )
            row = {
                "stage": stage_label,
                "epoch_or_step": int(epoch_or_step),
                "cumulative_training_online_secure_seconds": round(train_online, 6),
                "cumulative_instrumented_wall_seconds": round(
                    elapsed_offset + time.monotonic() - started, 6
                ),
                "evaluation_online_secure_seconds": round(
                    timer.seconds.get("stage_b_evaluation_online_secure", 0.0), 6
                ),
                "evaluation_reveal_seconds": round(
                    timer.seconds.get("stage_b_evaluation_reveal", 0.0), 6
                ),
                "development_accuracy": round(acc, 6),
                "epsilon_so_far": None,
                "epsilon_scope": "not_applicable_non_dp_stage_b",
                "used_for_training": False,
            }
            return row

        # Epoch 0 point (initialized model).
        if start_epoch == 0 and not args.no_stage_b_eval:
            trace_rows.append(stage_b_evaluate(0, "stage_b_epoch0", ledger, timer, spu))
            append_trace(stage_b_trace, trace_rows)

        for epoch in range(start_epoch, effective["stage_b_local_epochs"]):
            for batch_start in range(0, batches, chunk_batches):
                current_chunk_batches = min(chunk_batches, batches - batch_start)
                with timer.measure("stage_b_private_pack_and_infeed"):
                    packed = pack_exec(
                        data["b_cache_path"], data["cache_mode"], local_source_ids,
                        client_indices, epoch, args.batch_size, feature_dim,
                        num_classes, args.seed, batch_start, current_chunk_batches,
                    )
                    x_s = alice(lambda pair: pair[0])(packed).to(spu)
                    y_s = alice(lambda pair: pair[1])(packed).to(spu)
                    sf.wait([x_s, y_s])
                with timer.measure("stage_b_online_secure"):
                    client_w_s, client_b_s = epoch_exec(x_s, y_s, client_w_s, client_b_s)
                    ledger.wait(
                        "stage_b_epoch_{}_chunk_{}".format(epoch + 1, batch_start),
                        [client_w_s, client_b_s], "client_local_secret_state",
                    )
            if int(bcfg["fedavg_sync_interval"]) > 0 and (
                (epoch + 1) % int(bcfg["fedavg_sync_interval"]) == 0
            ):
                with timer.measure("stage_b_online_secure"):
                    client_w_s, client_b_s = sync_exec(client_w_s, client_b_s)
                    ledger.wait(
                        "stage_b_epoch_{}_fedavg_sync".format(epoch + 1),
                        [client_w_s, client_b_s], "client_local_secret_state",
                    )
            with timer.measure("stage_b_secret_checkpoint"):
                w_paths = dump_spu_object(
                    spu, client_w_s, checkpoint_dir, "current", "client_w"
                )
                b_paths = dump_spu_object(
                    spu, client_b_s, checkpoint_dir, "current", "client_b"
                )
                atomic_json(stage_b_meta, {
                    "epochs_done": epoch + 1,
                    "client_w_share_paths": w_paths,
                    "client_b_share_paths": b_paths,
                    "timing_seconds": timer.seconds,
                    "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
                })
            if (epoch + 1) % eval_interval == 0 and not args.no_stage_b_eval:
                trace_rows.append(
                    stage_b_evaluate(epoch + 1, "stage_b_epoch{}".format(epoch + 1), ledger, timer, spu)
                )
                append_trace(stage_b_trace, trace_rows)
            print(
                "[StageB trajectory] epoch={}/{}".format(epoch + 1, effective["stage_b_local_epochs"]),
                flush=True,
            )

        if not stage_b_complete:
            with timer.measure("stage_b_online_secure"):
                global_w_s, global_b_s = average_exec(client_w_s, client_b_s)
                ledger.wait(
                    "stage_b_final_average", [global_w_s, global_b_s], "global_secret_state"
                )
        if config["anchor_ce"] and not stage_b_complete:
            anchor_exec = spu_multi_return(
                spu, make_secure_anchor_ce(
                    config["anchor_steps"], config["anchor_lr"], config["anchor_mu"], args.seed
                )
            )
            with timer.measure("stage_b_anchor_infeed"):
                anchor_x_s = alice(lambda value=anchor_x.astype(np.float32): value)().to(spu)
                anchor_y_s = alice(lambda value=one_hot(anchor_y, num_classes): value)().to(spu)
                sf.wait([anchor_x_s, anchor_y_s])
            with timer.measure("stage_b_anchor_compile_warmup"):
                aw_s, ab_s = anchor_exec(anchor_x_s, anchor_y_s, global_w_s, global_b_s)
                sf.wait([aw_s, ab_s])
            # Pre-anchor point (uses the averaged head before refinement).
            if not stage_b_complete and not args.no_stage_b_eval:
                pre_anchor_row = stage_b_evaluate(
                    effective["stage_b_local_epochs"], "stage_b_pre_anchor", ledger, timer, spu
                )
                trace_rows.append(pre_anchor_row)
                append_trace(stage_b_trace, trace_rows)
            with timer.measure("stage_b_online_secure"):
                global_w_s, global_b_s = anchor_exec(anchor_x_s, anchor_y_s, global_w_s, global_b_s)
                ledger.wait(
                    "stage_b_anchor_ce", [global_w_s, global_b_s], "global_secret_state"
                )
            # Post-anchor point.
            if not args.no_stage_b_eval:
                post_anchor_row = stage_b_evaluate(
                    effective["stage_b_local_epochs"], "stage_b_post_anchor", ledger, timer, spu
                )
                trace_rows.append(post_anchor_row)
                append_trace(stage_b_trace, trace_rows)
        with timer.measure("stage_b_secret_checkpoint"):
            if stage_b_complete:
                global_w_paths = state["global_w_share_paths"]
                global_b_paths = state["global_b_share_paths"]
            else:
                global_w_paths = dump_spu_object(
                    spu, global_w_s, checkpoint_dir, "stage_b", "global_w"
                )
                global_b_paths = dump_spu_object(
                    spu, global_b_s, checkpoint_dir, "stage_b", "global_b"
                )
                atomic_json(stage_b_meta, {
                    "epochs_done": effective["stage_b_local_epochs"],
                    "stage_b_complete": True,
                    "global_w_share_paths": global_w_paths,
                    "global_b_share_paths": global_b_paths,
                    "timing_seconds": timer.seconds,
                    "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
                })

        # Expected Stage-B point count audit (pre-registered), computed once
        # here so both the stage_b_only gate report and the full-run audit use
        # the same number.
        if args.no_stage_b_eval:
            expected_b_points = 0
        else:
            expected_b_points = 1 + effective["stage_b_local_epochs"] // eval_interval
            if config["anchor_ce"]:
                expected_b_points += 2  # pre-anchor + post-anchor

        if args.stage_b_only:
            report.update({
                "status": "COMPLETE",
                "final_development_accuracy": (
                    float(trace_rows[-1]["development_accuracy"])
                    if trace_rows and not args.no_stage_b_eval
                    else None
                ),
                "conditional_stage_c_epsilon": None,
                "delta": 1e-5,
                "timing_seconds": timer.seconds,
                "online_secure_seconds": timer.seconds.get("stage_b_online_secure", 0.0),
                "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
                "resource_measurement": cgroup_delta(resource_start, cgroup_snapshot(), 32),
                "reveal_ledger": ledger.rows,
                "provider_metadata": {},
                "stage_b_client_state_public_reveals": 0,
                "stage_b_trace_points": len(trace_rows),
                "stage_b_expected_points": expected_b_points,
                "stage_c_trace_points": 0,
                "evaluation_scope": "aggregate development logits only; used_for_training=false",
                "checkpoint_kind": "per-party SPU secret shares",
                "checkpoint_paths": {"stage_b_w": global_w_paths, "stage_b_b": global_b_paths},
            })
            report["resource_measurement"]["run_observed_memory_peak_gib"] = memory_monitor.stop()
            atomic_json(report_path, report)
            print(json.dumps(report, sort_keys=True), flush=True)
            return

        # ----- Stage C -----
        total_steps = effective["stage_c_steps"]
        block_steps = int(args.block_steps)
        if total_steps % block_steps:
            raise ValueError("full-trajectory requires block_steps to divide total steps")
        capacity = int(stage_c["graph_capacity"])
        expected_batch = int(stage_c["expected_batch"])
        stage_c_exec = spu_multi_return(
            spu, corrected_stage_c_function(stage_c, expected_batch, block_steps)
        )
        with timer.measure("stage_c_compile_warmup"):
            z_x = np.zeros((block_steps, capacity, feature_dim), np.float32)
            z_y = np.zeros((block_steps, capacity, num_classes), np.float32)
            z_m = np.zeros((block_steps, capacity), np.float32)
            z_w = np.zeros((block_steps, feature_dim, num_classes), np.float32)
            z_b = np.zeros((block_steps, num_classes), np.float32)
            z_c = np.zeros((block_steps,), np.float32)
            warm = stage_c_exec(
                z_x, z_y, z_m, z_w, z_w, z_w, z_b, z_b, z_b,
                z_c, z_c, z_c, z_c, global_w_s, global_b_s, np.float32(stage_c["clip_c"]),
            )
            sf.wait(warm)

        steps_done = 0
        clip_s = np.float32(stage_c["clip_c"])
        stage_c_trace = output_dir / "stage_c_trace.csv"
        sc_rows = []
        if args.resume and state_meta.exists():
            state = json.loads(state_meta.read_text(encoding="utf-8"))
            if state.get("timing_seconds"):
                merge_prior_timing(timer, state["timing_seconds"])
                elapsed_offset = float(state.get("raw_wall_seconds", 0.0))
            steps_done = int(state["steps_done"])
            global_w_s = load_spu_object(spu, state["global_w_share_paths"])
            global_b_s = load_spu_object(spu, state["global_b_share_paths"])
            clip_s = load_spu_object(spu, state["clip_share_paths"])
            if stage_c_trace.exists():
                with stage_c_trace.open(newline="", encoding="utf-8") as handle:
                    sc_rows = list(csv.DictReader(handle))
            ledger.wait(
                "stage_c_resume", [global_w_s, global_b_s, clip_s], "dp_secret_state"
            )

        provider_paths = {"weight": None, "bias": None, "count": None}
        while steps_done < total_steps:
            block_ids = [
                poisson_batch_indices(len(data["c_x"]), expected_batch, random_bytes=os.urandom)
                for _ in range(block_steps)
            ]
            x_seq = np.zeros((block_steps, capacity, feature_dim), np.float32)
            y_seq = np.zeros((block_steps, capacity, num_classes), np.float32)
            mask_seq = np.zeros((block_steps, capacity), np.float32)
            for offset, ids in enumerate(block_ids):
                if len(ids) > capacity:
                    raise RuntimeError("Poisson draw exceeds graph capacity")
                x_seq[offset, :len(ids)] = np.asarray(data["c_x"])[ids]
                y_seq[offset, :len(ids)] = one_hot(np.asarray(data["c_y"])[ids], num_classes)
                mask_seq[offset, :len(ids)] = 1.0
            adaptive = np.asarray([
                float(step >= int(stage_c["warmup_steps"]) and (step + 1) % int(stage_c["update_interval"]) == 0)
                for step in range(steps_done, steps_done + block_steps)
            ], np.float32)
            with timer.measure("stage_c_private_infeed_and_noise_generation"):
                x_s = alice(lambda value=x_seq: value)().to(spu)
                y_s = alice(lambda value=y_seq: value)().to(spu)
                m_s = alice(lambda value=mask_seq: value)().to(spu)
                a_s = alice(lambda value=adaptive: value)().to(spu)
                nw_s, provider_paths["weight"] = request_private_noise_shares(
                    pyus, spu, CKPT / "private_party_noise_state", run_id,
                    "weight", (block_steps, feature_dim, num_classes),
                )
                nb_s, provider_paths["bias"] = request_private_noise_shares(
                    pyus, spu, CKPT / "private_party_noise_state", run_id,
                    "bias", (block_steps, num_classes),
                )
                nc_s, provider_paths["count"] = request_private_noise_shares(
                    pyus, spu, CKPT / "private_party_noise_state", run_id,
                    "count", (block_steps,),
                )
                sf.wait([x_s, y_s, m_s, a_s, nw_s, nb_s, nc_s])
            with timer.measure("stage_c_online_secure"):
                global_w_s, global_b_s, clip_s = stage_c_exec(
                    x_s, y_s, m_s, nw_s[0], nw_s[1], nw_s[2],
                    nb_s[0], nb_s[1], nb_s[2], nc_s[0], nc_s[1], nc_s[2],
                    a_s, global_w_s, global_b_s, clip_s,
                )
                ledger.wait(
                    "stage_c_steps_{}".format(steps_done + block_steps),
                    [global_w_s, global_b_s, clip_s], "dp_secret_state",
                )
            steps_done += block_steps
            with timer.measure("stage_c_secret_checkpoint"):
                w_paths = dump_spu_object(spu, global_w_s, checkpoint_dir, "stage_c", "global_w")
                b_paths = dump_spu_object(spu, global_b_s, checkpoint_dir, "stage_c", "global_b")
                c_paths = dump_spu_object(spu, clip_s, checkpoint_dir, "stage_c", "clip")
                atomic_json(state_meta, {
                    "steps_done": steps_done,
                    "global_w_share_paths": w_paths,
                    "global_b_share_paths": b_paths,
                    "clip_share_paths": c_paths,
                    "timing_seconds": timer.seconds,
                    "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
                })
            with timer.measure("stage_c_authorized_dp_release_and_dev_evaluation"):
                public_w, public_b, public_c = ledger.reveal(
                    "stage_c_global_release_{}".format(steps_done),
                    [global_w_s, global_b_s, clip_s], "conditional_dp_global_state",
                    "DP trajectory release for development-only utility trace",
                )
                dev_accuracy = float(np.mean(
                    np.argmax(np.asarray(data["c_x_dev"]) @ public_w + public_b, axis=1)
                    == np.asarray(data["c_y_dev"])
                ))
            with timer.measure("stage_c_privacy_accounting"):
                epsilon = conditional_stage_c_epsilon(stage_c, steps_done, len(data["c_x"]))
            sc_rows.append({
                "stage": "stage_c",
                "epoch_or_step": steps_done,
                "cumulative_training_online_secure_seconds": round(
                    timer.seconds.get("stage_b_online_secure", 0.0)
                    + timer.seconds.get("stage_c_online_secure", 0.0), 6
                ),
                "cumulative_instrumented_wall_seconds": round(
                    elapsed_offset + time.monotonic() - started, 6
                ),
                "evaluation_online_secure_seconds": 0.0,
                "evaluation_reveal_seconds": 0.0,
                "development_accuracy": round(dev_accuracy, 6),
                "clip_bound": round(float(np.asarray(public_c).reshape(-1)[0]), 6),
                "conditional_stage_c_epsilon": round(epsilon, 8),
                "delta": 1e-5,
                "used_for_training": False,
            })
            append_trace(stage_c_trace, sc_rows)
            atomic_json(state_meta, {
                "steps_done": steps_done,
                "global_w_share_paths": w_paths,
                "global_b_share_paths": b_paths,
                "clip_share_paths": c_paths,
                "timing_seconds": timer.seconds,
                "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
            })
            print(
                "[StageC trajectory] steps={}/{} acc={:.4f} eps={:.4f}".format(
                    steps_done, total_steps, dev_accuracy, epsilon
                ),
                flush=True,
            )

        metadata = {
            stream: reveal_private_noise_metadata(pyus, paths, ledger, stream)
            for stream, paths in provider_paths.items()
        }
        final_epsilon = conditional_stage_c_epsilon(stage_c, total_steps, len(data["c_x"]))
        if final_epsilon > 3.0 + 1e-6:
            raise AssertionError("conditional Stage-C epsilon exceeds target: {}".format(final_epsilon))

        # Combined trajectory for audit/plot. Unify column sets so the CSV
        # writer sees every field from both the Stage-B and Stage-C rows.
        combined = []
        for row in list(trace_rows) + list(sc_rows):
            base = {
                "stage": None, "epoch_or_step": None,
                "cumulative_training_online_secure_seconds": None,
                "cumulative_instrumented_wall_seconds": None,
                "evaluation_online_secure_seconds": None,
                "evaluation_reveal_seconds": None,
                "development_accuracy": None,
                "epsilon_so_far": None, "epsilon_scope": None,
                "clip_bound": None, "conditional_stage_c_epsilon": None,
                "delta": None, "used_for_training": None,
            }
            base.update(row)
            combined.append(base)
        combined_path = output_dir / "new_spu_full_trajectory_trace.csv"
        append_trace(combined_path, combined)

        actual_b_points = len(trace_rows)
        if actual_b_points != expected_b_points:
            raise AssertionError(
                "Stage-B trace points {} != expected {}".format(
                    actual_b_points, expected_b_points
                )
            )

        report.update({
            "status": "COMPLETE",
            "final_development_accuracy": float(sc_rows[-1]["development_accuracy"]),
            "conditional_stage_c_epsilon": final_epsilon,
            "delta": 1e-5,
            "timing_seconds": timer.seconds,
            "online_secure_seconds": (
                timer.seconds.get("stage_b_online_secure", 0.0)
                + timer.seconds.get("stage_c_online_secure", 0.0)
            ),
            "raw_wall_seconds": elapsed_offset + time.monotonic() - started,
            "resource_measurement": cgroup_delta(resource_start, cgroup_snapshot(), 32),
            "reveal_ledger": ledger.rows,
            "provider_metadata": metadata,
            "stage_b_client_state_public_reveals": 0,
            "stage_b_trace_points": actual_b_points,
            "stage_b_expected_points": expected_b_points,
            "stage_c_trace_points": len(sc_rows),
            "evaluation_scope": "aggregate development logits only; used_for_training=false",
            "checkpoint_kind": "per-party SPU secret shares",
            "checkpoint_paths": {"stage_b_w": global_w_paths, "stage_b_b": global_b_paths},
        })
        report["resource_measurement"]["run_observed_memory_peak_gib"] = memory_monitor.stop()
        report["resource_measurement"]["memory_peak_note"] = (
            "run_observed peak is authoritative; cgroup v1 lifetime max may predate this run"
        )
        atomic_json(report_path, report)
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
