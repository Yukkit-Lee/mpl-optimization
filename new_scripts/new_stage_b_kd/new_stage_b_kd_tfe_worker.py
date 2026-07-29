#!/usr/bin/env python3
"""Legacy Python 3.6 TFE/ABY3 worker for selected seed42 Stage-B KD.

All local MTT features, anchor features, labels, local heads, teacher
probabilities, reliability values, and student parameters remain secret-shared
until the explicit final correctness/reporting boundary.
"""

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
        weights = []
        biases = []
        for client_id in range(num_clients):
            x_clients.append(
                tfe.define_private_variable(
                    data["x_client_{}".format(client_id)],
                    name="stage_b_kd_client_x_{:02d}".format(client_id),
                )
            )
            y_clients.append(
                tfe.define_private_variable(
                    one_hot(
                        data["y_client_{}".format(client_id)], num_classes
                    ),
                    name="stage_b_kd_client_y_{:02d}".format(client_id),
                )
            )
            weights.append(
                tfe.define_private_variable(
                    data["initial_weight"],
                    name="stage_b_kd_local_weight_{:02d}".format(client_id),
                )
            )
            biases.append(
                tfe.define_private_variable(
                    data["initial_bias"],
                    name="stage_b_kd_local_bias_{:02d}".format(client_id),
                )
            )

        local_update_ops = []
        for client_id in range(num_clients):
            count = float(len(data["y_client_{}".format(client_id)]))
            logits = (
                secure_chunked_matmul(
                    tfe, x_clients[client_id], weights[client_id], chunk
                )
                + biases[client_id]
            )
            probabilities = tfe.softmax(logits)
            error = probabilities - y_clients[client_id]
            grad_weight = secure_chunked_matmul(
                tfe, tfe.transpose(x_clients[client_id]), error, chunk
            ) * (1.0 / count)
            grad_weight = (
                grad_weight
                + float(config["weight_decay"]) * weights[client_id]
            )
            grad_bias = tfe.reduce_sum(error, axis=0) * (1.0 / count)
            local_update_ops.append(
                secure_simultaneous_assign(
                    tf,
                    [
                        (
                            weights[client_id],
                            weights[client_id]
                            - float(config["local_lr"]) * grad_weight,
                        ),
                        (
                            biases[client_id],
                            biases[client_id]
                            - float(config["local_lr"]) * grad_bias,
                        ),
                    ],
                    "stage_b_kd_local_update_{:02d}".format(client_id),
                )
            )

        average_weight = weights[0] * (1.0 / num_clients)
        average_bias = biases[0] * (1.0 / num_clients)
        for client_id in range(1, num_clients):
            average_weight = (
                average_weight + weights[client_id] * (1.0 / num_clients)
            )
            average_bias = (
                average_bias + biases[client_id] * (1.0 / num_clients)
            )

        x_anchor = tfe.define_private_variable(
            data["x_anchor"], name="stage_b_kd_anchor_x"
        )
        y_anchor = tfe.define_private_variable(
            one_hot(data["y_anchor"], num_classes),
            name="stage_b_kd_anchor_y",
        )
        client_anchor_probabilities = []
        for client_id in range(num_clients):
            logits = (
                secure_chunked_matmul(
                    tfe, x_anchor, weights[client_id], chunk
                )
                + biases[client_id]
            )
            client_anchor_probabilities.append(
                tfe.softmax(logits * (1.0 / float(config["teacher_temperature"])))
            )

        if config["reliability_rule"] == "uniform":
            class_weights = None
            teacher = client_anchor_probabilities[0] * (1.0 / num_clients)
            for client_id in range(1, num_clients):
                teacher = (
                    teacher
                    + client_anchor_probabilities[client_id]
                    * (1.0 / num_clients)
                )
        elif config["reliability_rule"] == "confidence":
            reliability_rows = []
            for probabilities in client_anchor_probabilities:
                reliability_rows.append(
                    tfe.reduce_sum(
                        probabilities * y_anchor, axis=0
                    )
                    * (1.0 / float(config["anchor_samples_per_class"]))
                )
            reliability_client_class = tfe.stack(reliability_rows, axis=0)
            reliability_class_client = tfe.transpose(
                reliability_client_class
            )
            class_client_weights = tfe.softmax(
                reliability_class_client
                * (1.0 / float(config["weight_temperature"]))
            )
            class_weights = tfe.transpose(class_client_weights)
            teacher = client_anchor_probabilities[0] * class_weights[0]
            for client_id in range(1, num_clients):
                teacher = (
                    teacher
                    + client_anchor_probabilities[client_id]
                    * class_weights[client_id]
                )
            normalizer = tfe.reduce_sum(teacher, axis=1, keepdims=True)
            teacher = teacher * normalizer.reciprocal(nonsigned=True)
        else:
            raise ValueError(
                "TFE worker only accepts selected uniform/confidence rules"
            )

        student_weight = tfe.define_private_variable(
            np.zeros((feature_dim, num_classes), dtype=np.float32),
            name="stage_b_kd_student_weight",
        )
        student_bias = tfe.define_private_variable(
            np.zeros((num_classes,), dtype=np.float32),
            name="stage_b_kd_student_bias",
        )
        student_initialize = secure_simultaneous_assign(
            tf,
            [
                (student_weight, average_weight),
                (student_bias, average_bias),
            ],
            "stage_b_kd_student_initialize_from_average",
        )
        student_logits = (
            secure_chunked_matmul(
                tfe, x_anchor, student_weight, chunk
            )
            + student_bias
        )
        student_probabilities = tfe.softmax(student_logits)
        student_temperature_probabilities = tfe.softmax(
            student_logits
            * (1.0 / float(config["teacher_temperature"]))
        )
        combined_error = (
            (1.0 - float(config["kd_lambda"]))
            * (student_probabilities - y_anchor)
            + float(config["kd_lambda"])
            * float(config["teacher_temperature"])
            * (student_temperature_probabilities - teacher)
        )
        anchor_count = float(len(data["y_anchor"]))
        student_grad_weight = secure_chunked_matmul(
            tfe, tfe.transpose(x_anchor), combined_error, chunk
        ) * (1.0 / anchor_count)
        student_grad_bias = tfe.reduce_sum(
            combined_error, axis=0
        ) * (1.0 / anchor_count)
        if float(config["average_regularization"]) > 0.0:
            student_grad_weight = (
                student_grad_weight
                + 2.0
                * float(config["average_regularization"])
                * (student_weight - average_weight)
            )
            student_grad_bias = (
                student_grad_bias
                + 2.0
                * float(config["average_regularization"])
                * (student_bias - average_bias)
            )
        student_update = secure_simultaneous_assign(
            tf,
            [
                (
                    student_weight,
                    student_weight
                    - float(config["student_lr"]) * student_grad_weight,
                ),
                (
                    student_bias,
                    student_bias
                    - float(config["student_lr"]) * student_grad_bias,
                ),
            ],
            "stage_b_kd_student_update",
        )

        graph.update(
            {
                "local_update_ops": local_update_ops,
                "average_weight": average_weight,
                "average_bias": average_bias,
                "teacher": teacher,
                "class_weights": class_weights,
                "student_weight": student_weight,
                "student_bias": student_bias,
                "student_initialize": student_initialize,
                "student_update": student_update,
                "saver": tf.compat.v1.train.Saver(max_to_keep=2),
            }
        )
    return graph


def state_path(checkpoint_prefix):
    return checkpoint_prefix + "_state.json"


def save_state(graph, session, checkpoint_prefix, state):
    Path(checkpoint_prefix).parent.mkdir(parents=True, exist_ok=True)
    graph["saver"].save(session, checkpoint_prefix)
    atomic_json(state_path(checkpoint_prefix), state)


def load_state(graph, session, checkpoint_prefix):
    path = state_path(checkpoint_prefix)
    if not os.path.isfile(path) or not os.path.isfile(checkpoint_prefix + ".index"):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    graph["saver"].restore(session, checkpoint_prefix)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_prefix", required=True)
    parser.add_argument("--tiny_correctness", action="store_true")
    args = parser.parse_args()
    input_path = Path(args.input)
    with np.load(str(input_path), allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    config = json.loads(str(data["configuration_json"].reshape(-1)[0]))
    if config["seed"] != 42:
        raise ValueError("TFE worker is seed42-only")
    if config["initializer_source"] != "official_mtt_only":
        raise ValueError("TFE input lineage failed")
    if config["maximum_client_weight"] < 1.0 and not config[
        "cap_was_inactive_in_plaintext"
    ]:
        raise ValueError("secure cap implementation required but unavailable")
    if args.tiny_correctness:
        config = dict(config)
        config["local_epochs"] = 1
        config["student_steps"] = 1

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
            "schema": "new_stage_b_kd_tfe_state_v1",
            "local_epochs_completed": 0,
            "student_initialized": False,
            "student_steps_completed": 0,
            "local_secure_seconds": 0.0,
            "student_secure_seconds": 0.0,
        }
    checkpoint_interval = int(config["checkpoint_interval"])
    for epoch in range(
        int(state["local_epochs_completed"]), int(config["local_epochs"])
    ):
        started = time.time()
        for operation in graph["local_update_ops"]:
            session.run(operation)
        state["local_secure_seconds"] += time.time() - started
        state["local_epochs_completed"] = epoch + 1
        print(
            "[TFE-local] dataset={} epoch={}/{}".format(
                config["dataset"], epoch + 1, config["local_epochs"]
            ),
            flush=True,
        )
        if (epoch + 1) % checkpoint_interval == 0:
            save_state(graph, session, args.checkpoint_prefix, state)

    if not state["student_initialized"]:
        session.run(graph["student_initialize"])
        state["student_initialized"] = True
        save_state(graph, session, args.checkpoint_prefix, state)

    for step in range(
        int(state["student_steps_completed"]), int(config["student_steps"])
    ):
        started = time.time()
        session.run(graph["student_update"])
        state["student_secure_seconds"] += time.time() - started
        state["student_steps_completed"] = step + 1
        if (step + 1) in (1, int(config["student_steps"])) or (
            step + 1
        ) % checkpoint_interval == 0:
            print(
                "[TFE-student] dataset={} step={}/{}".format(
                    config["dataset"], step + 1, config["student_steps"]
                ),
                flush=True,
            )
        if (step + 1) % checkpoint_interval == 0:
            save_state(graph, session, args.checkpoint_prefix, state)

    fetches = [
        graph["average_weight"].reveal(),
        graph["average_bias"].reveal(),
        graph["teacher"].reveal(),
        graph["student_weight"].reveal(),
        graph["student_bias"].reveal(),
    ]
    if graph["class_weights"] is not None:
        fetches.append(graph["class_weights"].reveal())
    values = session.run(fetches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / (
        "new_stage_b_kd_tfe_model_{}_seed42.npz".format(config["dataset"])
    )
    arrays = {
        "average_weight": np.asarray(values[0], dtype=np.float32),
        "average_bias": np.asarray(values[1], dtype=np.float32),
        "teacher_anchor_probabilities": np.asarray(values[2], dtype=np.float32),
        "student_weight": np.asarray(values[3], dtype=np.float32),
        "student_bias": np.asarray(values[4], dtype=np.float32),
        "class_weights": (
            np.full(
                (int(config["num_clients"]), int(config["num_classes"])),
                1.0 / float(config["num_clients"]),
                dtype=np.float32,
            )
            if graph["class_weights"] is None
            else np.asarray(values[5], dtype=np.float32)
        ),
        "initializer_source": np.asarray(["official_mtt_only"]),
        "full_data_initializer_dependency": np.asarray([0], dtype=np.int64),
    }
    atomic_npz(model_path, **arrays)
    state["completed"] = True
    save_state(graph, session, args.checkpoint_prefix, state)
    peak_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
        1024.0 * 1024.0
    )
    report = {
        "schema": "new_stage_b_kd_tfe_report_v1",
        "status": "COMPLETE",
        "dataset": config["dataset"],
        "seed": 42,
        "backend": "TFE",
        "protocol": "ABY3",
        "fixedpoint_fractional": config["fixedpoint_fractional"],
        "tiny_correctness": bool(args.tiny_correctness),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "configuration": config,
        "graph_build_seconds": graph_build_seconds,
        "local_secure_seconds": state["local_secure_seconds"],
        "student_secure_seconds": state["student_secure_seconds"],
        "total_secure_seconds": (
            state["local_secure_seconds"] + state["student_secure_seconds"]
        ),
        "peak_rss_gib": peak_rss_gib,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "revealed_only_at_final_boundary": [
            "secure_average_weight_bias",
            "teacher_anchor_probabilities",
            "class_weights",
            "student_weight_bias",
        ],
        "no_plaintext_fallback": True,
        "custom_ops_required": True,
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "stage_c_run": False,
    }
    report_path = output_dir / (
        "new_stage_b_kd_tfe_report_{}_seed42.json".format(config["dataset"])
    )
    atomic_json(report_path, report)
    graph["KE"].clear_session()
    print(
        "COMPLETE report={} model={} secure_seconds={:.3f}".format(
            report_path, model_path, report["total_secure_seconds"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
