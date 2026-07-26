#!/usr/bin/env python3
# 用途：在 TFE/ABY3 中对 2048 维特征执行 Fixed/Adaptive DP-SGD。

from __future__ import print_function

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from new_stage_b_secure_init import atomic_json_dump
from new_stage_b_secure_init import atomic_npz_save
from new_stage_b_secure_init import import_tfe_or_fail
from new_stage_b_secure_init import load_feature_cache
from new_stage_b_secure_init import make_protocol
from new_stage_b_secure_init import one_hot
from new_stage_b_secure_init import secure_chunked_matmul
from new_stage_b_secure_init import secure_simultaneous_assign
from new_stage_b_secure_init import sha256_file


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODERN_PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON",
    os.path.join(os.path.dirname(PROJECT_ROOT), "new_envs", "distillation", "bin", "python"),
)
ACCOUNTANT_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "new_privacy_accountant.py"
)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def load_initializer(path):
    with np.load(path, allow_pickle=False) as cache:
        weight = np.asarray(cache["weight"], dtype=np.float32)
        bias = np.asarray(cache["bias"], dtype=np.float32)
    return weight, bias


def deterministic_batch_indices(num_samples, batch_size, seed, step):
    rng = np.random.RandomState(int(seed) + int(step) * 104729)
    if num_samples >= batch_size:
        return rng.choice(num_samples, size=batch_size, replace=False).astype(np.int64)
    return rng.choice(num_samples, size=batch_size, replace=True).astype(np.int64)


def _secure_uniform_float(tf, secure_random, seed, shape, nonce):
    """由 checkpointed CSPRNG seed 生成开区间 (0,1) 均匀数。"""
    maximum = 2147483647
    values = secure_random.seeded_random_uniform(
        shape,
        seed=seed + tf.cast(nonce, tf.int32),
        minval=1,
        maxval=maximum,
        dtype=tf.int64,
    )
    return (tf.cast(values, tf.float32) + 0.5) / float(maximum)


def _provider_gaussian(tf, secure_random, seed, shape, nonce, stddev):
    """Box-Muller：每个逻辑计算方只生成并持有自己的 Gaussian share。"""
    u1 = _secure_uniform_float(tf, secure_random, seed, shape, nonce)
    u2 = _secure_uniform_float(tf, secure_random, seed, shape, nonce + 1)
    radius = tf.sqrt(-2.0 * tf.log(u1))
    return radius * tf.cos(2.0 * math.pi * u2) * float(stddev)


def distributed_standard_gaussian(
    tf, tfe, protocol, secure_random, shape, seeds, counters, name
):
    """
    三方独立生成 N(0,1/3) share，再在 ABY3 中求和得到 N(0,1)。

    seed 与 counter 均是各方本地 checkpoint 变量；不会由单个 NumPy 采样器
    生成总噪声后再伪装成 MPC noise。
    """
    private_parts = []
    for party_id in range(3):
        provider = "server{}".format(party_id)

        def provide(pid=party_id):
            with tf.device(protocol.servers[pid].device_name):
                offset = {"weight": 0, "bias": 2, "count": 4}[name]
                nonce = counters[pid] * 8 + offset
                return _provider_gaussian(
                    tf,
                    secure_random,
                    seeds[pid],
                    shape,
                    nonce,
                    1.0 / math.sqrt(3.0),
                )

        private_parts.append(
            tfe.define_private_input(provider, provide, name="{}_noise_p{}".format(name, party_id))
        )
    return private_parts[0] + private_parts[1] + private_parts[2]


def secure_clip_factor(tfe, norm_squared, clip_bound):
    """计算 min(1, C/sqrt(norm_squared))，不 reveal per-example norm。"""
    raw_factor = clip_bound * tfe.inv_sqrt(norm_squared + 1e-5)
    return 1.0 - tfe.maximum(1.0 - raw_factor, 0.0)


def secure_clamp(tfe, value, lower, upper):
    value = tfe.maximum(value, float(lower))
    return -tfe.maximum(-value, -float(upper))


def adaptive_clip_direction_reference(current_c, fraction_below, gamma, eta_c):
    """Plain reference for testing the direction of the secure C update."""
    return float(current_c) * math.exp(
        -float(eta_c) * (float(fraction_below) - float(gamma))
    )


def released_model_dp_valid(initializer_privacy_status):
    return initializer_privacy_status in (
        "data_independent",
        "disjoint_public",
        "dp_sanitized",
    )


def build_graph(data, init_weight, init_bias, args):
    tf, tfe, KE, FixedpointConfig = import_tfe_or_fail()
    from tf_encrypted.operations.secure_random import secure_random

    tf.compat.v1.reset_default_graph()
    protocol = make_protocol(tfe, FixedpointConfig, args.precision_fractional)
    graph = {"tf": tf, "tfe": tfe, "KE": KE, "protocol": protocol}

    with protocol:
        with tf.device(protocol.servers[0].device_name):
            raw_x = tf.compat.v1.placeholder(
                tf.float32, [args.batch_size, args.feature_dim], name="new_stage_c_x"
            )
            raw_y = tf.compat.v1.placeholder(
                tf.float32, [args.batch_size, args.num_classes], name="new_stage_c_y"
            )
        private_x = tfe.define_private_input("server0", lambda: raw_x)
        private_y = tfe.define_private_input("server0", lambda: raw_y)
        weight = tfe.define_private_variable(init_weight, name="new_stage_c_weight")
        bias = tfe.define_private_variable(init_bias, name="new_stage_c_bias")
        clip_bound = tfe.define_private_variable(
            np.asarray(float(args.clip_c), dtype=np.float32), name="new_clip_bound"
        )

        # 每方持有独立 CSPRNG seed；Saver 会同时保存 seed 与 counter。
        provider_seeds = []
        provider_counters = []
        for party_id in range(3):
            with tf.device(protocol.servers[party_id].device_name):
                provider_seeds.append(
                    tf.Variable(
                        secure_random.secure_seed(),
                        trainable=False,
                        dtype=tf.int32,
                        name="new_noise_seed_p{}".format(party_id),
                    )
                )
                provider_counters.append(
                    tf.Variable(
                        0,
                        trainable=False,
                        dtype=tf.int32,
                        name="new_noise_counter_p{}".format(party_id),
                    )
                )
        standard_noise_w = distributed_standard_gaussian(
            tf,
            tfe,
            protocol,
            secure_random,
            [args.feature_dim, args.num_classes],
            provider_seeds,
            provider_counters,
            "weight",
        )
        standard_noise_b = distributed_standard_gaussian(
            tf,
            tfe,
            protocol,
            secure_random,
            [args.num_classes],
            provider_seeds,
            provider_counters,
            "bias",
        )
        standard_noise_count = distributed_standard_gaussian(
            tf,
            tfe,
            protocol,
            secure_random,
            [],
            provider_seeds,
            provider_counters,
            "count",
        )

        logits = secure_chunked_matmul(
            tfe, private_x, weight, args.matmul_chunk_size
        ) + bias
        error = logits - private_y
        error_sq = tfe.reduce_sum(error * error, axis=1)
        feature_sq = tfe.reduce_sum(private_x * private_x, axis=1)
        weight_norm_squared = error_sq * feature_sq
        bias_norm_squared = error_sq
        norm_squared = weight_norm_squared + bias_norm_squared
        if args.clipping_scope == "global":
            factor_w = secure_clip_factor(tfe, norm_squared, clip_bound)
            factor_b = factor_w
            # 1 means the example is below C and therefore is not clipped.
            below = tfe.select(
                tfe.less_equal(norm_squared, clip_bound * clip_bound), 1.0, 0.0
            )
        else:
            factor_w = secure_clip_factor(tfe, weight_norm_squared, clip_bound)
            factor_b = secure_clip_factor(tfe, bias_norm_squared, clip_bound)
            below_w = tfe.select(
                tfe.less_equal(weight_norm_squared, clip_bound * clip_bound), 1.0, 0.0
            )
            below_b = tfe.select(
                tfe.less_equal(bias_norm_squared, clip_bound * clip_bound), 1.0, 0.0
            )
            below = (below_w + below_b) * 0.5
        clipped_error_w = error * tfe.expand_dims(factor_w, axis=1)
        clipped_error_b = error * tfe.expand_dims(factor_b, axis=1)
        clipped_grad_w = secure_chunked_matmul(
            tfe,
            tfe.transpose(private_x),
            clipped_error_w,
            args.matmul_chunk_size,
        ) * (1.0 / float(args.batch_size))
        clipped_grad_b = tfe.reduce_sum(clipped_error_b, axis=0) * (
            1.0 / float(args.batch_size)
        )

        noise_scale = float(args.noise_multiplier) * clip_bound / float(
            args.batch_size
        )
        noisy_grad_w = clipped_grad_w + standard_noise_w * noise_scale
        noisy_grad_b = clipped_grad_b + standard_noise_b * noise_scale
        new_weight = weight - float(args.learning_rate) * noisy_grad_w
        new_bias = bias - float(args.learning_rate) * noisy_grad_b

        fraction_below = tfe.reduce_sum(below) * (1.0 / float(args.batch_size))
        noisy_fraction = secure_clamp(
            tfe,
            fraction_below
            + standard_noise_count
            * (float(args.count_noise_multiplier) / float(args.batch_size)),
            0.0,
            1.0,
        )
        exponent = -float(args.eta_c) * (noisy_fraction - float(args.gamma))
        candidate_c = clip_bound * tfe.exp(exponent)
        new_clip_bound = secure_clamp(tfe, candidate_c, args.c_min, args.c_max)

        fixed_update = secure_simultaneous_assign(
            tf,
            [(weight, new_weight), (bias, new_bias)],
            "new_stage_c_fixed_atomic_update",
        )
        adaptive_update_core = secure_simultaneous_assign(
            tf,
            [
                (weight, new_weight),
                (bias, new_bias),
                (clip_bound, new_clip_bound),
            ],
            "new_stage_c_adaptive_atomic_update",
        )

        def with_counter_increment(core_op, scope):
            with tf.name_scope(scope):
                with tf.control_dependencies([core_op]):
                    increments = [
                        tf.compat.v1.assign_add(value, 1)
                        for value in provider_counters
                    ]
                return tf.group(*increments)

        fixed_update = with_counter_increment(
            fixed_update, "new_stage_c_fixed_counter_commit"
        )
        adaptive_update = with_counter_increment(
            adaptive_update_core, "new_stage_c_adaptive_counter_commit"
        )

        reference_x = tfe.define_private_variable(data["x_reference"])
        reference_y = tfe.define_private_variable(
            one_hot(data["y_reference"], args.num_classes)
        )
        reference_logits = secure_chunked_matmul(
            tfe, reference_x, weight, args.matmul_chunk_size
        ) + bias
        reference_loss = tfe.reduce_sum(
            (reference_logits - reference_y) * (reference_logits - reference_y)
        ) * (1.0 / float(len(data["x_reference"]) * args.num_classes))

        graph.update(
            {
                "raw_x": raw_x,
                "raw_y": raw_y,
                "weight": weight,
                "bias": bias,
                "clip_bound": clip_bound,
                "logits": logits,
                "norm_squared": norm_squared,
                "factor": factor_w,
                "factor_w": factor_w,
                "factor_b": factor_b,
                "clipped_grad_w": clipped_grad_w,
                "clipped_grad_b": clipped_grad_b,
                "standard_noise_w": standard_noise_w,
                "standard_noise_b": standard_noise_b,
                "fraction_below": fraction_below,
                "noisy_fraction": noisy_fraction,
                "new_clip_bound": new_clip_bound,
                "fixed_update": fixed_update,
                "adaptive_update": adaptive_update,
                "reference_logits": reference_logits,
                "reference_loss": reference_loss,
                "provider_counters": provider_counters,
                "saver": tf.compat.v1.train.Saver(max_to_keep=2),
            }
        )
    return graph


def adaptive_update_count(steps, warmup_steps, update_interval):
    return sum(
        1
        for step in range(int(steps))
        if step >= int(warmup_steps)
        and (step + 1) % int(update_interval) == 0
    )


def accountant_epsilon(steps, sample_rate, args):
    command = [
        MODERN_PYTHON,
        ACCOUNTANT_SCRIPT,
        "--steps",
        str(int(steps)),
        "--sample_rate",
        str(float(sample_rate)),
        "--noise_multiplier",
        str(float(args.noise_multiplier)),
        "--delta",
        str(float(args.delta)),
    ]
    if args.mode == "adaptive_dp":
        command.extend(
            [
                "--count_noise",
                str(float(args.count_noise_multiplier)),
                "--count_steps",
                str(
                    adaptive_update_count(
                        steps, args.warmup_steps, args.update_interval
                    )
                ),
            ]
        )
    output = subprocess.check_output(command, universal_newlines=True)
    return float(json.loads(output)["epsilon"])


def checkpoint_state_path(prefix):
    return prefix + "_new_state.json"


def save_checkpoint(graph, session, prefix, state):
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    graph["saver"].save(session, prefix)
    atomic_json_dump(state, checkpoint_state_path(prefix))


def load_checkpoint(graph, session, prefix):
    state_path = checkpoint_state_path(prefix)
    if not os.path.exists(state_path) or not os.path.exists(prefix + ".index"):
        return None
    graph["saver"].restore(session, prefix)
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_secure_stage_c(data, init_weight, init_bias, args):
    graph = build_graph(data, init_weight, init_bias, args)
    tf, tfe, KE = graph["tf"], graph["tfe"], graph["KE"]
    session = tfe.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    state = load_checkpoint(graph, session, args.checkpoint_prefix)
    if state is None:
        state = {
            "schema": "new_secure_stage_c_checkpoint_v1",
            "steps_completed": 0,
            "consecutive_cmax_hits": 0,
            "history": [],
        }
    history = list(state.get("history", []))
    log_steps = set(
        np.linspace(1, args.steps, min(10, args.steps)).round().astype(np.int64).tolist()
    )
    started = time.time()
    for step in range(int(state["steps_completed"]), int(args.steps)):
        ids = deterministic_batch_indices(
            len(data["x_encoder_train"]), args.batch_size, args.seed, step
        )
        feed = {
            graph["raw_x"]: data["x_encoder_train"][ids],
            graph["raw_y"]: one_hot(data["y_encoder_train"][ids], args.num_classes),
        }
        use_adaptive = (
            args.mode == "adaptive_dp"
            and step >= int(args.warmup_steps)
            and (step + 1) % int(args.update_interval) == 0
        )
        update = graph["adaptive_update"] if use_adaptive else graph["fixed_update"]
        current_c = float(
            np.asarray(session.run(graph["clip_bound"].reveal())).reshape(-1)[0]
        )
        if use_adaptive:
            fetched = session.run(
                [
                    update,
                    graph["noisy_fraction"].reveal(),
                    graph["new_clip_bound"].reveal(),
                ],
                feed_dict=feed,
            )
            noisy_fraction_below = float(
                np.asarray(fetched[1]).reshape(-1)[0]
            )
            proposed_c = float(np.asarray(fetched[2]).reshape(-1)[0])
        else:
            session.run(update, feed_dict=feed)
            noisy_fraction_below = None
            proposed_c = current_c
        state["steps_completed"] = step + 1
        row = {
            "step": step + 1,
            "clipping_bound_t": current_c,
            "proposed_next_c": proposed_c,
            "adaptive_clip_update": bool(use_adaptive),
            "dp_noised_fraction_below": noisy_fraction_below,
            "dp_noised_clipped_fraction": (
                None
                if noisy_fraction_below is None
                else 1.0 - noisy_fraction_below
            ),
        }
        history.append(row)
        state["history"] = history
        if use_adaptive and proposed_c >= float(args.c_max) * 0.999:
            state["consecutive_cmax_hits"] = (
                int(state.get("consecutive_cmax_hits", 0)) + 1
            )
        else:
            state["consecutive_cmax_hits"] = 0
        if int(state["consecutive_cmax_hits"]) >= int(args.exploding_c_patience):
            save_checkpoint(graph, session, args.checkpoint_prefix, state)
            raise RuntimeError(
                "Adaptive C_t 连续 {} 次撞击 C_max={}，已安全停止".format(
                    state["consecutive_cmax_hits"], args.c_max
                )
            )
        if step + 1 in log_steps:
            ref_loss, ref_logits = session.run(
                [graph["reference_loss"].reveal(), graph["reference_logits"].reveal()]
            )
            ref_acc = float(
                np.mean(
                    np.argmax(ref_logits, axis=1)
                    == np.asarray(data["y_reference"], dtype=np.int64)
                )
            )
            row["reference_loss"] = float(np.asarray(ref_loss).reshape(-1)[0])
            row["reference_accuracy"] = ref_acc
            print(
                "[StageC] step={}/{} ref_acc={:.4f} C_t={:.5f} noisy_b={}".format(
                    step + 1,
                    args.steps,
                    ref_acc,
                    current_c,
                    (
                        "N/A"
                        if row["dp_noised_fraction_below"] is None
                        else "{:.4f}".format(
                            row["dp_noised_fraction_below"]
                        )
                    ),
                ),
                flush=True,
            )
        if (step + 1) % int(args.checkpoint_interval) == 0:
            save_checkpoint(graph, session, args.checkpoint_prefix, state)

    final_weight, final_bias, final_c, counters = session.run(
        [
            graph["weight"].reveal(),
            graph["bias"].reveal(),
            graph["clip_bound"].reveal(),
            graph["provider_counters"],
        ]
    )
    save_checkpoint(graph, session, args.checkpoint_prefix, state)
    runtime = time.time() - started
    KE.clear_session()
    return {
        "weight": np.asarray(final_weight, dtype=np.float32),
        "bias": np.asarray(final_bias, dtype=np.float32),
        "final_c": float(np.asarray(final_c).reshape(-1)[0]),
        "provider_counters": [int(value) for value in counters],
        "history": history,
        "runtime_seconds": float(runtime),
        "steps_completed": int(state["steps_completed"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TFE/ABY3 Secure Stage C DP-SGD")
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny"], required=True)
    parser.add_argument("--mode", choices=["fixed_dp", "adaptive_dp"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--noise_multiplier", type=float, default=1.2)
    parser.add_argument("--count_noise_multiplier", type=float, default=8.0)
    parser.add_argument("--clip_c", type=float, default=1.0)
    parser.add_argument(
        "--clipping_scope",
        choices=["global", "parameter_block"],
        default="global",
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta_c", type=float, default=0.05)
    parser.add_argument("--c_min", type=float, default=0.05)
    parser.add_argument("--c_max", type=float, default=4.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--update_interval", type=int, default=5)
    parser.add_argument("--exploding_c_patience", type=int, default=5)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--precision_fractional", type=int, default=16)
    parser.add_argument("--matmul_chunk_size", type=int, default=64)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument(
        "--initializer_privacy_status",
        choices=[
            "data_independent",
            "disjoint_public",
            "dp_sanitized",
            "data_dependent_non_dp",
        ],
        default="data_dependent_non_dp",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_feature_cache(args.feature_cache)
    init_weight, init_bias = load_initializer(args.initializer)
    args.feature_dim = int(data["feature_dim"])
    args.num_classes = int(init_bias.shape[0])
    if args.feature_dim != 2048 or init_weight.shape != (
        args.feature_dim,
        args.num_classes,
    ):
        raise ValueError("Stage C 仅接受与 initializer 对齐的 2048 维特征")
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 92, flush=True)
    print("Stage C：TFE/ABY3 Secure {} DP-SGD".format(args.mode), flush=True)
    print(
        "dataset={} samples={} batch={} steps={} feature_dim={}".format(
            args.dataset,
            len(data["x_encoder_train"]),
            args.batch_size,
            args.steps,
            args.feature_dim,
        ),
        flush=True,
    )
    print(
        "noise_multiplier={} count_noise={} delta={} cpu_policy=unrestricted".format(
            args.noise_multiplier, args.count_noise_multiplier, args.delta
        ),
        flush=True,
    )
    print("=" * 92, flush=True)
    result = run_secure_stage_c(data, init_weight, init_bias, args)
    sample_rate = min(1.0, float(args.batch_size) / len(data["x_encoder_train"]))
    epsilon = accountant_epsilon(result["steps_completed"], sample_rate, args)
    count_steps = (
        adaptive_update_count(
            result["steps_completed"], args.warmup_steps, args.update_interval
        )
        if args.mode == "adaptive_dp"
        else 0
    )
    model_path = os.path.join(
        args.output_dir,
        "new_secure_stage_c_model_{}_{}_seed{}.npz".format(
            args.dataset, args.mode, args.seed
        ),
    )
    atomic_npz_save(
        model_path,
        weight=result["weight"],
        bias=result["bias"],
        clipping_bound=np.asarray([result["final_c"]], dtype=np.float32),
    )
    final_model_dp_valid = released_model_dp_valid(
        args.initializer_privacy_status
    )
    report = {
        "schema": "new_secure_stage_c_report_v2",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": args.seed,
        "mode": args.mode,
        "feature_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "configuration": vars(args),
        "privacy": {
            "accountant": "Opacus RDPAccountant",
            "accounted_gradient_mechanism_epsilon": epsilon,
            "stage_c_distilled_feature_record_epsilon": epsilon,
            "stage_c_delta": args.delta,
            "sample_rate": sample_rate,
            "steps": result["steps_completed"],
            "gradient_noise_multiplier": args.noise_multiplier,
            "clipping_count_noise_multiplier": args.count_noise_multiplier,
            "clipping_count_mechanism_steps": count_steps,
            "initializer_privacy_status": args.initializer_privacy_status,
            "released_model_record_dp_valid": final_model_dp_valid,
            "epsilon_interpretation": (
                "released_model_record_dp"
                if final_model_dp_valid
                else "conditional_gradient_mechanism_only_not_released_model_dp"
            ),
            "stage_a_original_record_privacy": "inherited_from_stage_a_metadata",
            "epsilon_a_plus_epsilon_c_claimed": False,
        },
        "security": {
            "backend": "tfe_mpc",
            "protocol": "ABY3",
            "execution": "single-machine process simulation",
            "noise_generation": "three_independent_checkpointed_csprng_gaussian_shares",
            "secret_shared": [
                "features",
                "labels",
                "parameters",
                "per_example_norms",
                "clip_factors",
                "gradient_sums",
                "noise_shares_after_input_sharing",
                "updated_parameters",
                "adaptive_clipping_bound",
            ],
            "revealed": [
                (
                    "dp_noised_fraction_below_at_adaptive_updates"
                    if args.mode == "adaptive_dp"
                    else "no_clipping_count_statistic"
                ),
                (
                    "dp_adaptive_clipping_bound_trajectory"
                    if args.mode == "adaptive_dp"
                    else "fixed_public_clipping_bound"
                ),
                "aggregate_reference_logits_at_log_intervals",
                (
                    "final_dp_model"
                    if final_model_dp_valid
                    else "final_noisy_model_without_end_to_end_dp_claim"
                ),
            ],
            "silent_plaintext_fallback": False,
            "honest_party_assumption": "at_least_one_of_three_compute_parties",
        },
        "final_c": result["final_c"],
        "provider_counters": result["provider_counters"],
        "history": result["history"],
        "runtime_seconds": result["runtime_seconds"],
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "limitations": [
            "Reference logits are revealed only for correctness/tuning diagnostics and must be disabled or reduced in the frozen formal release.",
            "TFE local mode is a single-machine process simulation; LAN/WAN communication is not measured.",
        ],
    }
    report_path = os.path.join(
        args.output_dir,
        "new_secure_stage_c_report_{}_{}_seed{}.json".format(
            args.dataset, args.mode, args.seed
        ),
    )
    atomic_json_dump(report, report_path)
    print("epsilon_C={:.6f} final_C={:.6f}".format(epsilon, result["final_c"]))
    print("model={}".format(model_path))
    print("report={}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
