#!/usr/bin/env python3
"""Legacy Python 3.6 TFE/ABY3 worker for frozen C1/C4/C5 Stage B."""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_stage_b_secure_init import (  # noqa: E402
    import_tfe_or_fail,
    make_protocol,
    secure_chunked_matmul,
    secure_simultaneous_assign,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def one_hot(labels, num_classes):
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return result


def build_graph(data, config):
    tf, tfe, KE, FixedpointConfig = import_tfe_or_fail()
    tf.compat.v1.reset_default_graph()
    protocol = make_protocol(
        tfe, FixedpointConfig, int(config["fixedpoint_fractional"])
    )
    graph = {"tf": tf, "tfe": tfe, "KE": KE, "protocol": protocol}
    num_clients = int(config["num_clients"])
    num_classes = int(config["num_classes"])
    feature_dim = int(config["feature_dim"])
    chunk = int(config["matmul_chunk_size"])
    with protocol:
        x_clients = []
        y_clients = []
        local_weights = []
        local_biases = []
        for client_id in range(num_clients):
            x_clients.append(
                tfe.define_private_variable(
                    data["x_client_{}".format(client_id)],
                    name="postagg_client_x_{:02d}".format(client_id),
                )
            )
            y_clients.append(
                tfe.define_private_variable(
                    one_hot(
                        data["y_client_{}".format(client_id)], num_classes
                    ),
                    name="postagg_client_y_{:02d}".format(client_id),
                )
            )
            local_weights.append(
                tfe.define_private_variable(
                    data["initial_weight"],
                    name="postagg_local_weight_{:02d}".format(client_id),
                )
            )
            local_biases.append(
                tfe.define_private_variable(
                    data["initial_bias"],
                    name="postagg_local_bias_{:02d}".format(client_id),
                )
            )
        local_updates = []
        for client_id in range(num_clients):
            count = float(len(data["y_client_{}".format(client_id)]))
            logits = (
                secure_chunked_matmul(
                    tfe,
                    x_clients[client_id],
                    local_weights[client_id],
                    chunk,
                )
                + local_biases[client_id]
            )
            probabilities = tfe.softmax(logits)
            error = probabilities - y_clients[client_id]
            grad_weight = secure_chunked_matmul(
                tfe, tfe.transpose(x_clients[client_id]), error, chunk
            ) * (1.0 / count)
            grad_weight = (
                grad_weight
                + float(config["weight_decay"]) * local_weights[client_id]
            )
            grad_bias = tfe.reduce_sum(error, axis=0) * (1.0 / count)
            local_updates.append(
                secure_simultaneous_assign(
                    tf,
                    [
                        (
                            local_weights[client_id],
                            local_weights[client_id]
                            - float(config["local_lr"]) * grad_weight,
                        ),
                        (
                            local_biases[client_id],
                            local_biases[client_id]
                            - float(config["local_lr"]) * grad_bias,
                        ),
                    ],
                    "postagg_local_update_{:02d}".format(client_id),
                )
            )

        average_weight = local_weights[0] * (1.0 / num_clients)
        average_bias = local_biases[0] * (1.0 / num_clients)
        for client_id in range(1, num_clients):
            average_weight = (
                average_weight
                + local_weights[client_id] * (1.0 / num_clients)
            )
            average_bias = (
                average_bias
                + local_biases[client_id] * (1.0 / num_clients)
            )

        x_anchor = tfe.define_private_variable(
            data["x_anchor"], name="postagg_anchor_x"
        )
        y_anchor = tfe.define_private_variable(
            one_hot(data["y_anchor"], num_classes),
            name="postagg_anchor_y",
        )
        random_reference_weight = tfe.define_private_variable(
            data["initial_weight"], name="postagg_random_reference_weight"
        )
        random_reference_bias = tfe.define_private_variable(
            data["initial_bias"], name="postagg_random_reference_bias"
        )
        c1_weight = tfe.define_private_variable(
            data["initial_weight"], name="postagg_c1_weight"
        )
        c1_bias = tfe.define_private_variable(
            data["initial_bias"], name="postagg_c1_bias"
        )
        c5_weight = tfe.define_private_variable(
            np.zeros((feature_dim, num_classes), dtype=np.float32),
            name="postagg_c5_weight",
        )
        c5_bias = tfe.define_private_variable(
            np.zeros((num_classes,), dtype=np.float32),
            name="postagg_c5_bias",
        )
        c5_initialize = secure_simultaneous_assign(
            tf,
            [(c5_weight, average_weight), (c5_bias, average_bias)],
            "postagg_c5_initialize_from_secure_average",
        )

        anchor_count = float(len(data["y_anchor"]))

        def anchor_update(weight, bias, reference_weight, reference_bias, name):
            logits = (
                secure_chunked_matmul(tfe, x_anchor, weight, chunk) + bias
            )
            probabilities = tfe.softmax(logits)
            error = probabilities - y_anchor
            grad_weight = secure_chunked_matmul(
                tfe, tfe.transpose(x_anchor), error, chunk
            ) * (1.0 / anchor_count)
            grad_bias = tfe.reduce_sum(error, axis=0) * (
                1.0 / anchor_count
            )
            if float(config["anchor_mu"]) > 0.0:
                grad_weight = (
                    grad_weight
                    + 2.0
                    * float(config["anchor_mu"])
                    * (weight - reference_weight)
                )
                grad_bias = (
                    grad_bias
                    + 2.0
                    * float(config["anchor_mu"])
                    * (bias - reference_bias)
                )
            return secure_simultaneous_assign(
                tf,
                [
                    (
                        weight,
                        weight - float(config["anchor_lr"]) * grad_weight,
                    ),
                    (
                        bias,
                        bias - float(config["anchor_lr"]) * grad_bias,
                    ),
                ],
                name,
            )

        c1_update = anchor_update(
            c1_weight,
            c1_bias,
            random_reference_weight,
            random_reference_bias,
            "postagg_c1_anchor_update",
        )
        c5_update = anchor_update(
            c5_weight,
            c5_bias,
            average_weight,
            average_bias,
            "postagg_c5_anchor_update",
        )
        graph.update(
            {
                "local_updates": local_updates,
                "average_weight": average_weight,
                "average_bias": average_bias,
                "c5_initialize": c5_initialize,
                "c1_weight": c1_weight,
                "c1_bias": c1_bias,
                "c5_weight": c5_weight,
                "c5_bias": c5_bias,
                "c1_update": c1_update,
                "c5_update": c5_update,
                "saver": tf.compat.v1.train.Saver(max_to_keep=2),
            }
        )
    return graph


def state_path(prefix):
    return prefix + "_state.json"


def save_state(graph, session, prefix, state):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    graph["saver"].save(session, prefix)
    atomic_json(state_path(prefix), state)


def load_state(graph, session, prefix):
    if not os.path.isfile(state_path(prefix)) or not os.path.isfile(
        prefix + ".index"
    ):
        return None
    with open(state_path(prefix), "r", encoding="utf-8") as handle:
        state = json.load(handle)
    graph["saver"].restore(session, prefix)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_prefix", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    with np.load(str(input_path), allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    config = json.loads(str(data["configuration_json"].reshape(-1)[0]))
    if config["seed"] != 42 or config["distribution"] != "severe":
        raise ValueError("worker only accepts frozen seed42 Severe")
    if config["initializer_source"] != "official_mtt_only":
        raise ValueError("Official-MTT lineage gate failed")
    graph_started = time.time()
    graph = build_graph(data, config)
    graph_build_seconds = time.time() - graph_started
    tf = graph["tf"]
    tfe = graph["tfe"]
    session = tfe.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    state = load_state(graph, session, args.checkpoint_prefix)
    if state is None:
        state = {
            "schema": "new_stage_b_anchor_postagg_tfe_state_v1",
            "local_epochs_completed": 0,
            "c5_initialized": False,
            "anchor_steps_completed": 0,
            "local_secure_seconds": 0.0,
            "secure_average_initialize_seconds": 0.0,
            "c1_anchor_secure_seconds": 0.0,
            "c5_anchor_secure_seconds": 0.0,
        }
    interval = int(config["checkpoint_interval"])
    for epoch in range(
        int(state["local_epochs_completed"]), int(config["local_epochs"])
    ):
        started = time.time()
        for operation in graph["local_updates"]:
            session.run(operation)
        state["local_secure_seconds"] += time.time() - started
        state["local_epochs_completed"] = epoch + 1
        print(
            "[TFE-local] {} {}/{}".format(
                config["dataset"], epoch + 1, config["local_epochs"]
            ),
            flush=True,
        )
        if (epoch + 1) % interval == 0:
            save_state(graph, session, args.checkpoint_prefix, state)
    if not state["c5_initialized"]:
        started = time.time()
        session.run(graph["c5_initialize"])
        state["secure_average_initialize_seconds"] += time.time() - started
        state["c5_initialized"] = True
        save_state(graph, session, args.checkpoint_prefix, state)
    for step in range(
        int(state["anchor_steps_completed"]), int(config["anchor_steps"])
    ):
        started = time.time()
        session.run(graph["c1_update"])
        state["c1_anchor_secure_seconds"] += time.time() - started
        started = time.time()
        session.run(graph["c5_update"])
        state["c5_anchor_secure_seconds"] += time.time() - started
        state["anchor_steps_completed"] = step + 1
        if (step + 1) in (1, int(config["anchor_steps"])) or (
            step + 1
        ) % interval == 0:
            print(
                "[TFE-anchor] {} {}/{}".format(
                    config["dataset"], step + 1, config["anchor_steps"]
                ),
                flush=True,
            )
        if (step + 1) % interval == 0:
            save_state(graph, session, args.checkpoint_prefix, state)
    values = session.run(
        [
            graph["average_weight"].reveal(),
            graph["average_bias"].reveal(),
            graph["c1_weight"].reveal(),
            graph["c1_bias"].reveal(),
            graph["c5_weight"].reveal(),
            graph["c5_bias"].reveal(),
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / (
        "new_stage_b_anchor_postagg_tfe_model_{}_seed42.npz".format(
            config["dataset"]
        )
    )
    atomic_npz(
        model_path,
        average_weight=np.asarray(values[0], dtype=np.float32),
        average_bias=np.asarray(values[1], dtype=np.float32),
        c1_weight=np.asarray(values[2], dtype=np.float32),
        c1_bias=np.asarray(values[3], dtype=np.float32),
        c5_weight=np.asarray(values[4], dtype=np.float32),
        c5_bias=np.asarray(values[5], dtype=np.float32),
        initializer_source=np.asarray(["official_mtt_only"]),
        full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
    )
    state["completed"] = True
    save_state(graph, session, args.checkpoint_prefix, state)
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
        1024.0 * 1024.0
    )
    total = (
        state["local_secure_seconds"]
        + state["secure_average_initialize_seconds"]
        + state["c1_anchor_secure_seconds"]
        + state["c5_anchor_secure_seconds"]
    )
    report = {
        "schema": "new_stage_b_anchor_postagg_tfe_report_v1",
        "status": "COMPLETE",
        "dataset": config["dataset"],
        "seed": 42,
        "backend": "TFE",
        "protocol": "ABY3",
        "fixedpoint_fractional": config["fixedpoint_fractional"],
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "configuration": config,
        "graph_build_seconds": graph_build_seconds,
        "local_secure_seconds": state["local_secure_seconds"],
        "secure_average_initialize_seconds": state[
            "secure_average_initialize_seconds"
        ],
        "c1_anchor_secure_seconds": state["c1_anchor_secure_seconds"],
        "c5_anchor_secure_seconds": state["c5_anchor_secure_seconds"],
        "total_secure_seconds": total,
        "peak_rss_gib": peak_rss,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "revealed_only_at_final_correctness_boundary": [
            "C4_secure_average_weight_bias",
            "C1_random_anchor_weight_bias",
            "C5_average_anchor_weight_bias",
        ],
        "secure_average_executed_under_secret_sharing": True,
        "anchor_data_and_labels_secret_shared": True,
        "no_plaintext_fallback": True,
        "custom_ops_required": True,
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
        "stage_c_run": False,
        "end_to_end_run": False,
    }
    report_path = output_dir / (
        "new_stage_b_anchor_postagg_tfe_report_{}_seed42.json".format(
            config["dataset"]
        )
    )
    atomic_json(report_path, report)
    graph["KE"].clear_session()
    print(
        "COMPLETE {} secure_seconds={:.3f}".format(
            report_path, total
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
