#!/usr/bin/env python3
# 用途：在 TF Encrypted/ABY3 中完成 Stage B 的 logit-MSE 蒸馏、候选聚合与私有候选选择。

from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_npz_save(path, **arrays):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.zeros((len(labels), int(num_classes)), dtype=np.float32)
    values[np.arange(len(labels)), labels] = 1.0
    return values


def load_feature_cache(path, mode="npz"):
    if mode == "memmap":
        # Memmap mode: features stay in .npy files, accessed via indices
        try:
            from new_full_feature_memmap_loader import load_feature_cache_memmap
        except ImportError:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            scripts_dir = os.path.join(script_dir, "new_scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from new_full_feature_memmap_loader import load_feature_cache_memmap
        result = load_feature_cache_memmap(path)
        # Ensure compatible types
        result["x_reference"] = result["x_reference"].astype(np.float32)
        result["x_selection"] = result["x_selection"].astype(np.float32)
        result["y_encoder_train"] = result["y_encoder_train"].astype(np.int64)
        result["y_reference"] = result["y_reference"].astype(np.int64)
        result["y_selection"] = result["y_selection"].astype(np.int64)
        return result

    # Default NPZ mode (backward compatible)
    with np.load(path, allow_pickle=False) as cache:
        required = [
            "x_encoder_train",
            "y_encoder_train",
            "x_reference",
            "y_reference",
            "x_selection",
            "y_selection",
            "feature_dim",
        ]
        missing = [name for name in required if name not in cache.files]
        if missing:
            raise ValueError("特征缓存缺少字段: {}".format(missing))
        result = {name: np.asarray(cache[name]) for name in required}
    result["x_encoder_train"] = result["x_encoder_train"].astype(np.float32)
    result["x_reference"] = result["x_reference"].astype(np.float32)
    result["x_selection"] = result["x_selection"].astype(np.float32)
    result["y_encoder_train"] = result["y_encoder_train"].astype(np.int64)
    result["y_reference"] = result["y_reference"].astype(np.int64)
    result["y_selection"] = result["y_selection"].astype(np.int64)
    result["feature_dim"] = int(result["feature_dim"].reshape(-1)[0])
    return result


def make_client_splits(labels, num_clients, split, seed):
    """构造可复现的 IID 或 Dirichlet non-IID 索引，且保证每个客户端非空。"""
    labels = np.asarray(labels, dtype=np.int64)
    num_clients = int(num_clients)
    rng = np.random.RandomState(int(seed))
    if split == "iid":
        indices = rng.permutation(len(labels))
        return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]
    if split == "balanced":
        buckets = [[] for _ in range(num_clients)]
        for class_id in np.unique(labels):
            class_indices = np.where(labels == class_id)[0]
            class_rng = np.random.RandomState(
                int(seed) + int(class_id) * 1009 + 728
            )
            class_indices = class_rng.permutation(class_indices)
            for client_id, part in enumerate(
                np.array_split(class_indices, num_clients)
            ):
                buckets[client_id].extend(part.tolist())
        return [
            np.random.RandomState(int(seed) + client_id * 7919).permutation(
                np.asarray(bucket, dtype=np.int64)
            )
            for client_id, bucket in enumerate(buckets)
        ]

    alpha = 1.0 if split == "mild" else 0.1
    classes = np.unique(labels)
    for retry in range(100):
        buckets = [[] for _ in range(num_clients)]
        retry_rng = np.random.RandomState(int(seed) + retry * 7919)
        for class_id in classes:
            class_indices = np.where(labels == class_id)[0]
            retry_rng.shuffle(class_indices)
            proportions = retry_rng.dirichlet(
                np.full(num_clients, alpha, dtype=np.float64)
            )
            counts = retry_rng.multinomial(len(class_indices), proportions)
            offset = 0
            for client_id, count in enumerate(counts):
                buckets[client_id].extend(
                    class_indices[offset : offset + count].tolist()
                )
                offset += count
        if min(len(bucket) for bucket in buckets) > 0:
            return [
                retry_rng.permutation(np.asarray(bucket, dtype=np.int64))
                for bucket in buckets
            ]
    raise RuntimeError("无法构造每个客户端均非空的 {} 划分".format(split))


def split_summary(labels, client_indices):
    summary = []
    for client_id, indices in enumerate(client_indices):
        unique, counts = np.unique(labels[indices], return_counts=True)
        summary.append(
            {
                "client_id": int(client_id),
                "num_samples": int(len(indices)),
                "class_histogram": {
                    str(int(key)): int(value) for key, value in zip(unique, counts)
                },
            }
        )
    return summary


def aggregation_weights(labels, client_indices, mode):
    """Return public aggregation coefficients derived only from split metadata."""
    if mode == "uniform":
        values = np.ones(len(client_indices), dtype=np.float64)
    elif mode == "sample_count":
        values = np.asarray(
            [len(indices) for indices in client_indices], dtype=np.float64
        )
    elif mode == "class_balanced":
        global_counts = np.bincount(np.asarray(labels, dtype=np.int64))
        values = np.asarray(
            [
                float(np.sum(1.0 / global_counts[labels[indices]]))
                for indices in client_indices
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("未知客户端聚合权重: {}".format(mode))
    return values / np.sum(values)


def deterministic_batches(indices, batch_size, seed, phase, epoch, client_id):
    """固定顺序生成满 batch；尾部通过本客户端样本循环补齐，避免改变静态图形状。"""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("客户端数据不能为空")
    rng = np.random.RandomState(
        int(seed) + int(phase) * 1000003 + int(epoch) * 1009 + int(client_id) * 97
    )
    shuffled = rng.permutation(indices)
    batch_count = int(np.ceil(float(len(shuffled)) / float(batch_size)))
    needed = batch_count * int(batch_size)
    if needed > len(shuffled):
        extra = np.resize(shuffled, needed - len(shuffled))
        shuffled = np.concatenate([shuffled, extra], axis=0)
    return [
        shuffled[offset : offset + int(batch_size)]
        for offset in range(0, len(shuffled), int(batch_size))
    ]


def initialize_heads(num_clients, feature_dim, num_classes, seed, scale):
    rng = np.random.RandomState(int(seed))
    weights = []
    biases = []
    for _ in range(int(num_clients)):
        weights.append(
            rng.normal(
                0.0, float(scale), size=(int(feature_dim), int(num_classes))
            ).astype(np.float32)
        )
        biases.append(np.zeros((int(num_classes),), dtype=np.float32))
    return weights, biases


def mse_loss(logits, targets):
    error = logits - targets
    return float(np.mean(error * error))


def plaintext_reference(
    x_train,
    y_train,
    x_reference,
    y_reference,
    x_selection,
    y_selection,
    client_indices,
    initial_weights,
    initial_biases,
    args,
):
    """与安全图使用相同更新顺序的 NumPy 参考实现，仅允许 correctness 使用。"""
    weights = [value.copy() for value in initial_weights]
    biases = [value.copy() for value in initial_biases]
    y_train_hot = one_hot(y_train, args.num_classes)
    y_reference_hot = one_hot(y_reference, args.num_classes)
    y_selection_hot = one_hot(y_selection, args.num_classes)
    coefficients = np.asarray(args.client_aggregation_weights, dtype=np.float64)
    for epoch in range(int(args.local_epochs)):
        for client_id, indices in enumerate(client_indices):
            for batch in deterministic_batches(
                indices, args.batch_size, args.seed, 1, epoch, client_id
            ):
                logits = np.matmul(x_train[batch], weights[client_id]) + biases[
                    client_id
                ]
                error = logits - y_train_hot[batch]
                grad_w = np.matmul(x_train[batch].T, error) / float(len(batch))
                grad_b = np.sum(error, axis=0) / float(len(batch))
                grad_w += float(args.weight_decay) * weights[client_id]
                weights[client_id] -= float(args.local_lr) * grad_w
                biases[client_id] -= float(args.local_lr) * grad_b
        if (
            int(args.fedavg_sync_interval) > 0
            and (epoch + 1) % int(args.fedavg_sync_interval) == 0
        ):
            avg_weight = np.sum(
                np.stack(weights, axis=0) * coefficients[:, None, None],
                axis=0,
            )
            avg_bias = np.sum(
                np.stack(biases, axis=0) * coefficients[:, None], axis=0
            )
            weights = [avg_weight.copy() for _ in weights]
            biases = [avg_bias.copy() for _ in biases]

    teacher = np.sum(
        np.stack(
            [
                np.matmul(x_reference, weight) + bias
                for weight, bias in zip(weights, biases)
            ],
            axis=0,
        )
        * coefficients[:, None, None],
        axis=0,
    )
    nokd_weights = [value.copy() for value in weights]
    nokd_biases = [value.copy() for value in biases]
    nokd_avg_w = np.sum(
        np.stack(nokd_weights, axis=0) * coefficients[:, None, None], axis=0
    )
    nokd_avg_b = np.sum(
        np.stack(nokd_biases, axis=0) * coefficients[:, None], axis=0
    )
    for epoch in range(int(args.distill_epochs)):
        for client_id, indices in enumerate(client_indices):
            for batch in deterministic_batches(
                indices, args.batch_size, args.seed, 2, epoch, client_id
            ):
                local_logits = (
                    np.matmul(x_train[batch], weights[client_id])
                    + biases[client_id]
                )
                local_error = local_logits - y_train_hot[batch]
                ref_logits = (
                    np.matmul(x_reference, weights[client_id]) + biases[client_id]
                )
                kd_error = ref_logits - teacher
                reference_error = ref_logits - y_reference_hot
                grad_w = np.matmul(x_train[batch].T, local_error) / float(
                    len(batch)
                )
                grad_w += float(args.lambda_kd) * np.matmul(
                    x_reference.T, kd_error
                ) / float(len(x_reference))
                reference_weight = float(
                    getattr(args, "reference_supervised_weight", 0.0)
                )
                if reference_weight > 0.0:
                    grad_w += reference_weight * np.matmul(
                        x_reference.T, reference_error
                    ) / float(len(x_reference))
                grad_b = np.sum(local_error, axis=0) / float(len(batch))
                grad_b += float(args.lambda_kd) * np.sum(kd_error, axis=0) / float(
                    len(x_reference)
                )
                if reference_weight > 0.0:
                    grad_b += reference_weight * np.sum(
                        reference_error, axis=0
                    ) / float(len(x_reference))
                grad_w += float(args.weight_decay) * weights[client_id]
                weights[client_id] -= float(args.distill_lr) * grad_w
                biases[client_id] -= float(args.distill_lr) * grad_b

    avg_w = np.sum(
        np.stack(weights, axis=0) * coefficients[:, None, None], axis=0
    )
    avg_b = np.sum(
        np.stack(biases, axis=0) * coefficients[:, None], axis=0
    )
    local_losses = np.asarray(
        [
            mse_loss(np.matmul(x_reference, weight) + bias, y_reference_hot)
            for weight, bias in zip(weights, biases)
        ],
        dtype=np.float64,
    )
    reliability = np.exp(-local_losses / float(args.weight_temperature))
    reliability /= np.sum(reliability)
    rw_w = np.sum(
        np.stack(weights, axis=0) * reliability[:, None, None], axis=0
    )
    rw_b = np.sum(np.stack(biases, axis=0) * reliability[:, None], axis=0)
    avg_score = mse_loss(np.matmul(x_selection, avg_w) + avg_b, y_selection_hot)
    rw_score = mse_loss(np.matmul(x_selection, rw_w) + rw_b, y_selection_hot)
    if rw_score <= avg_score:
        selected_w, selected_b, selected_name = rw_w, rw_b, "KD-ReferenceWeighted"
    else:
        selected_w, selected_b, selected_name = avg_w, avg_b, "KD-Avg"
    return {
        "nokd_weights": nokd_weights,
        "nokd_biases": nokd_biases,
        "nokd_avg_w": nokd_avg_w,
        "nokd_avg_b": nokd_avg_b,
        "weights": weights,
        "biases": biases,
        "teacher": teacher,
        "avg_w": avg_w,
        "avg_b": avg_b,
        "rw_w": rw_w,
        "rw_b": rw_b,
        "reference_weights": reliability,
        "avg_selection_mse": avg_score,
        "rw_selection_mse": rw_score,
        "selected_w": selected_w,
        "selected_b": selected_b,
        "selected_name": selected_name,
    }


def import_tfe_or_fail():
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    import tf_encrypted as tfe
    from tf_encrypted.keras import backend as KE
    from tf_encrypted.operations.aux import aux as aux_impl
    from tf_encrypted.operations.secure_random import secure_random as random_impl
    from tf_encrypted.tensor.fixed import FixedpointConfig

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if aux_impl.aux_module is None:
        raise RuntimeError("TFE aux custom op 未加载，禁止明文或慢速静默回退")
    if not random_impl.supports_secure_randomness():
        raise RuntimeError("TFE secure_random custom op 未加载，禁止不安全随机回退")
    return tf, tfe, KE, FixedpointConfig


def make_protocol(tfe, FixedpointConfig, precision_fractional):
    protocol = tfe.protocol.ABY3()
    if int(precision_fractional) == 16:
        return protocol
    protocol.fixedpoint_config = FixedpointConfig(
        scaling_base=2,
        precision_integral=11,
        precision_fractional=int(precision_fractional),
        # 2048 维内积必须分块截断，沿用旧版 TFE 已验证的安全阈值。
        matmul_threshold=256,
        truncation_gap=0,
        use_noninteractive_truncation=False,
    )
    return protocol


def secure_mse(tfe, logits, targets, count):
    error = logits - targets
    return tfe.reduce_sum(error * error) * (1.0 / float(count))


def secure_chunked_matmul(tfe, left, right, chunk_size):
    """
    对私有内积维度显式分块。

    TFE 0.8.0 的 ABY3 matmul 不读取 matmul_threshold；2048 维一次累加后
    才截断会破坏 int64 固定点数值，因此每个小块独立安全截断后再求和。
    """
    left_shape = left.shape.as_list()
    right_shape = right.shape.as_list()
    inner = int(left_shape[-1])
    if inner != int(right_shape[-2]):
        raise ValueError("matmul 维度不匹配: {} vs {}".format(left_shape, right_shape))
    chunk_size = int(chunk_size)
    if inner <= chunk_size:
        return tfe.matmul(left, right)
    result = None
    for start in range(0, inner, chunk_size):
        end = min(inner, start + chunk_size)
        partial = tfe.matmul(left[:, start:end], right[start:end, :])
        result = partial if result is None else result + partial
    return result


def secure_simultaneous_assign(tf, assignments, name):
    """
    原子更新一组相互依赖的 ABY3 私有变量。

    TFE 0.8.0 的单变量 assign 只等待该变量的新 shares。线性头的 W/b
    更新彼此依赖，若把两个 assign 直接并行执行，会出现读取旧变量与写入
    新变量交错的竞态。这里先物化全部新 shares，再统一执行底层 share 赋值；
    全程不 reconstruct、不 reveal。
    """
    dependencies = []
    for _, value in assignments:
        dependencies.extend(
            share.value for server_shares in value.unwrapped for share in server_shares
        )

    operations = []
    with tf.name_scope(name):
        with tf.control_dependencies(dependencies):
            for variable, value in assignments:
                for server_id in range(3):
                    for replica_id in range(2):
                        operations.append(
                            variable.unwrapped[server_id][replica_id].assign_from_same(
                                value.unwrapped[server_id][replica_id]
                            )
                        )
    return tf.group(*operations, name="commit_all_private_shares")


def build_secure_graph(data, client_indices, initial_weights, initial_biases, args):
    tf, tfe, KE, FixedpointConfig = import_tfe_or_fail()
    tf.compat.v1.reset_default_graph()
    protocol = make_protocol(tfe, FixedpointConfig, args.precision_fractional)
    graph = {"tf": tf, "tfe": tfe, "KE": KE, "protocol": protocol}

    with protocol:
        # 原生 placeholder 只位于输入拥有方；随后由 define_private_input 在图内
        # 使用 CSPRNG 拆分 ABY3 shares，禁止调用旧版 PrivatePlaceholder.feed()。
        with tf.device(protocol.servers[0].device_name):
            raw_batch_x = tf.compat.v1.placeholder(
                tf.float32,
                shape=[int(args.batch_size), int(args.feature_dim)],
                name="new_local_provider_batch_x",
            )
            raw_batch_y = tf.compat.v1.placeholder(
                tf.float32,
                shape=[int(args.batch_size), int(args.num_classes)],
                name="new_local_provider_batch_y",
            )
        batch_x = tfe.define_private_input(
            "server0", lambda: raw_batch_x, name="local_batch_x"
        )
        batch_y = tfe.define_private_input(
            "server0", lambda: raw_batch_y, name="local_batch_y"
        )
        reference_x = tfe.define_private_variable(
            data["x_reference"], name="reference_x"
        )
        reference_y = tfe.define_private_variable(
            one_hot(data["y_reference"], args.num_classes), name="reference_y"
        )
        selection_x = tfe.define_private_variable(
            data["x_selection"], name="selection_x"
        )
        selection_y = tfe.define_private_variable(
            one_hot(data["y_selection"], args.num_classes), name="selection_y"
        )
        weights = [
            tfe.define_private_variable(value, name="local_weight_{:02d}".format(i))
            for i, value in enumerate(initial_weights)
        ]
        biases = [
            tfe.define_private_variable(value, name="local_bias_{:02d}".format(i))
            for i, value in enumerate(initial_biases)
        ]

        local_ops = []
        for client_id in range(int(args.num_clients)):
            logits = secure_chunked_matmul(
                tfe, batch_x, weights[client_id], args.matmul_chunk_size
            ) + biases[client_id]
            error = logits - batch_y
            grad_w = secure_chunked_matmul(
                tfe, tfe.transpose(batch_x), error, args.matmul_chunk_size
            ) * (
                1.0 / float(args.batch_size)
            )
            grad_b = tfe.reduce_sum(error, axis=0) * (
                1.0 / float(args.batch_size)
            )
            if float(args.weight_decay) > 0.0:
                grad_w = grad_w + float(args.weight_decay) * weights[client_id]
            new_weight = weights[client_id] - float(args.local_lr) * grad_w
            new_bias = biases[client_id] - float(args.local_lr) * grad_b
            local_ops.append(
                secure_simultaneous_assign(
                    tf,
                    [
                        (weights[client_id], new_weight),
                        (biases[client_id], new_bias),
                    ],
                    "new_atomic_local_update_{:02d}".format(client_id),
                )
            )

        coefficients = [
            float(value) for value in args.client_aggregation_weights
        ]
        teacher_expression = None
        for coefficient, weight, bias in zip(coefficients, weights, biases):
            logits = secure_chunked_matmul(
                tfe, reference_x, weight, args.matmul_chunk_size
            ) + bias
            logits = logits * coefficient
            teacher_expression = (
                logits
                if teacher_expression is None
                else teacher_expression + logits
            )
        nokd_avg_w_expression = weights[0] * coefficients[0]
        nokd_avg_b_expression = biases[0] * coefficients[0]
        for client_id in range(1, int(args.num_clients)):
            nokd_avg_w_expression = (
                nokd_avg_w_expression
                + weights[client_id] * coefficients[client_id]
            )
            nokd_avg_b_expression = (
                nokd_avg_b_expression
                + biases[client_id] * coefficients[client_id]
            )
        fedavg_sync = secure_simultaneous_assign(
            tf,
            [
                (weights[client_id], nokd_avg_w_expression)
                for client_id in range(int(args.num_clients))
            ]
            + [
                (biases[client_id], nokd_avg_b_expression)
                for client_id in range(int(args.num_clients))
            ],
            "new_atomic_fedavg_sync",
        )
        teacher_logits = tfe.define_private_variable(
            np.zeros(
                (len(data["x_reference"]), int(args.num_classes)), dtype=np.float32
            ),
            name="teacher_logits",
        )
        nokd_avg_w = tfe.define_private_variable(
            np.zeros(
                (int(args.feature_dim), int(args.num_classes)),
                dtype=np.float32,
            ),
            name="nokd_avg_weight",
        )
        nokd_avg_b = tfe.define_private_variable(
            np.zeros((int(args.num_classes),), dtype=np.float32),
            name="nokd_avg_bias",
        )
        teacher_assign = secure_simultaneous_assign(
            tf,
            [
                (teacher_logits, teacher_expression),
                (nokd_avg_w, nokd_avg_w_expression),
                (nokd_avg_b, nokd_avg_b_expression),
            ],
            "new_atomic_teacher_and_nokd_snapshot",
        )

        distill_ops = []
        for client_id in range(int(args.num_clients)):
            local_logits = secure_chunked_matmul(
                tfe, batch_x, weights[client_id], args.matmul_chunk_size
            ) + biases[client_id]
            local_error = local_logits - batch_y
            ref_logits = (
                secure_chunked_matmul(
                    tfe, reference_x, weights[client_id], args.matmul_chunk_size
                )
                + biases[client_id]
            )
            kd_error = ref_logits - teacher_logits
            reference_error = ref_logits - reference_y
            grad_w = secure_chunked_matmul(
                tfe,
                tfe.transpose(batch_x),
                local_error,
                args.matmul_chunk_size,
            ) * (
                1.0 / float(args.batch_size)
            )
            grad_w = grad_w + float(args.lambda_kd) * secure_chunked_matmul(
                tfe,
                tfe.transpose(reference_x),
                kd_error,
                args.matmul_chunk_size,
            ) * (1.0 / float(len(data["x_reference"])))
            reference_weight = float(
                getattr(args, "reference_supervised_weight", 0.0)
            )
            if reference_weight > 0.0:
                grad_w = grad_w + reference_weight * secure_chunked_matmul(
                    tfe,
                    tfe.transpose(reference_x),
                    reference_error,
                    args.matmul_chunk_size,
                ) * (1.0 / float(len(data["x_reference"])))
            grad_b = tfe.reduce_sum(local_error, axis=0) * (
                1.0 / float(args.batch_size)
            )
            grad_b = grad_b + float(args.lambda_kd) * tfe.reduce_sum(
                kd_error, axis=0
            ) * (1.0 / float(len(data["x_reference"])))
            if reference_weight > 0.0:
                grad_b = grad_b + reference_weight * tfe.reduce_sum(
                    reference_error, axis=0
                ) * (1.0 / float(len(data["x_reference"])))
            if float(args.weight_decay) > 0.0:
                grad_w = grad_w + float(args.weight_decay) * weights[client_id]
            new_weight = weights[client_id] - float(args.distill_lr) * grad_w
            new_bias = biases[client_id] - float(args.distill_lr) * grad_b
            distill_ops.append(
                secure_simultaneous_assign(
                    tf,
                    [
                        (weights[client_id], new_weight),
                        (biases[client_id], new_bias),
                    ],
                    "new_atomic_distill_update_{:02d}".format(client_id),
                )
            )

        avg_w = weights[0] * coefficients[0]
        avg_b = biases[0] * coefficients[0]
        for client_id in range(1, int(args.num_clients)):
            avg_w = avg_w + weights[client_id] * coefficients[client_id]
            avg_b = avg_b + biases[client_id] * coefficients[client_id]

        # Softmax is invariant to a shared shift. Centering the private losses
        # keeps low-temperature exponentials inside the fixed-point range
        # without revealing any client loss or changing the aggregation rule.
        reference_losses = []
        for weight, bias in zip(weights, biases):
            logits = secure_chunked_matmul(
                tfe, reference_x, weight, args.matmul_chunk_size
            ) + bias
            loss = secure_mse(
                tfe,
                logits,
                reference_y,
                len(data["x_reference"]) * int(args.num_classes),
            )
            reference_losses.append(loss)
        reference_loss_mean = reference_losses[0]
        for loss in reference_losses[1:]:
            reference_loss_mean = reference_loss_mean + loss
        reference_loss_mean = reference_loss_mean * (
            1.0 / float(args.num_clients)
        )
        reliability = [
            tfe.exp(
                (loss - reference_loss_mean)
                * (-1.0 / float(args.weight_temperature))
            )
            for loss in reference_losses
        ]
        reliability_sum = reliability[0]
        for value in reliability[1:]:
            reliability_sum = reliability_sum + value
        reliability_inverse = reliability_sum.reciprocal(nonsigned=True)
        normalized_weights = [
            value * reliability_inverse for value in reliability
        ]

        rw_w = weights[0] * normalized_weights[0]
        rw_b = biases[0] * normalized_weights[0]
        for client_id in range(1, int(args.num_clients)):
            rw_w = rw_w + weights[client_id] * normalized_weights[client_id]
            rw_b = rw_b + biases[client_id] * normalized_weights[client_id]

        avg_selection_logits = secure_chunked_matmul(
            tfe, selection_x, avg_w, args.matmul_chunk_size
        ) + avg_b
        rw_selection_logits = secure_chunked_matmul(
            tfe, selection_x, rw_w, args.matmul_chunk_size
        ) + rw_b
        avg_selection_mse = secure_mse(
            tfe,
            avg_selection_logits,
            selection_y,
            len(data["x_selection"]) * int(args.num_classes),
        )
        rw_selection_mse = secure_mse(
            tfe,
            rw_selection_logits,
            selection_y,
            len(data["x_selection"]) * int(args.num_classes),
        )
        choose_rw = rw_selection_mse <= avg_selection_mse
        # 先把单个布尔选择位转为算术 share，再用标量广播完成私有 mux。
        choose_rw_arithmetic = tfe.b2a_single(choose_rw)
        selected_w = avg_w + choose_rw_arithmetic * (rw_w - avg_w)
        selected_b = avg_b + choose_rw_arithmetic * (rw_b - avg_b)

        graph.update(
            {
                "batch_x": batch_x,
                "batch_y": batch_y,
                "raw_batch_x": raw_batch_x,
                "raw_batch_y": raw_batch_y,
                "weights": weights,
                "biases": biases,
                "local_ops": local_ops,
                "fedavg_sync": fedavg_sync,
                "teacher_logits": teacher_logits,
                "teacher_assign": teacher_assign,
                "nokd_avg_w": nokd_avg_w,
                "nokd_avg_b": nokd_avg_b,
                "distill_ops": distill_ops,
                "avg_w": avg_w,
                "avg_b": avg_b,
                "rw_w": rw_w,
                "rw_b": rw_b,
                "normalized_weights": normalized_weights,
                "avg_selection_mse": avg_selection_mse,
                "rw_selection_mse": rw_selection_mse,
                "choose_rw": choose_rw,
                "selected_w": selected_w,
                "selected_b": selected_b,
                "saver": tf.compat.v1.train.Saver(max_to_keep=2),
            }
        )
    return graph


def checkpoint_state_path(prefix):
    return prefix + "_new_state.json"


def save_checkpoint(graph, session, prefix, state):
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    graph["saver"].save(session, prefix)
    atomic_json_dump(state, checkpoint_state_path(prefix))


def load_checkpoint_if_available(graph, session, prefix):
    state_path = checkpoint_state_path(prefix)
    index_path = prefix + ".index"
    if not os.path.exists(state_path) or not os.path.exists(index_path):
        return None
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    graph["saver"].restore(session, prefix)
    return state


def run_secure_stage_b(data, client_indices, initial_weights, initial_biases, args):
    graph = build_secure_graph(
        data, client_indices, initial_weights, initial_biases, args
    )
    tf, tfe, KE = graph["tf"], graph["tfe"], graph["KE"]
    # TFE 0.8.0 的 Session(config=...) 接收的是 TFE Config，不是 TF ConfigProto。
    # 不注入 TF/BLAS 单线程环境变量，由 AutoDL runtime 使用可用 CPU。
    session = tfe.Session()
    started = time.time()
    session.run(tf.compat.v1.global_variables_initializer())
    state = load_checkpoint_if_available(graph, session, args.checkpoint_prefix)
    if state is None:
        state = {
            "schema": "new_secure_stage_b_checkpoint_v1",
            "local_epochs_completed": 0,
            "teacher_initialized": False,
            "distill_epochs_completed": 0,
            "history": [],
            "elapsed_seconds_completed": 0.0,
        }
    else:
        print(
            "恢复 Stage B checkpoint: local={} teacher={} distill={}".format(
                state["local_epochs_completed"],
                state["teacher_initialized"],
                state["distill_epochs_completed"],
            ),
            flush=True,
        )
    state.setdefault("history", [])
    state.setdefault("elapsed_seconds_completed", 0.0)
    history = list(state["history"])
    elapsed_seconds = float(state["elapsed_seconds_completed"])

    log_epochs = set(
        np.linspace(1, max(1, int(args.local_epochs)), min(10, args.local_epochs))
        .round()
        .astype(np.int64)
        .tolist()
    )
    for epoch in range(
        int(state["local_epochs_completed"]), int(args.local_epochs)
    ):
        epoch_started = time.time()
        epoch_batches = 0
        for client_id, indices in enumerate(client_indices):
            for batch in deterministic_batches(
                indices, args.batch_size, args.seed, 1, epoch, client_id
            ):
                feed = {
                    graph["raw_batch_x"]: data["x_encoder_train"][batch],
                    graph["raw_batch_y"]: one_hot(
                        data["y_encoder_train"][batch], args.num_classes
                    ),
                }
                session.run(graph["local_ops"][client_id], feed_dict=feed)
                epoch_batches += 1
        synchronized = (
            int(args.fedavg_sync_interval) > 0
            and (epoch + 1) % int(args.fedavg_sync_interval) == 0
        )
        if synchronized:
            session.run(graph["fedavg_sync"])
        epoch_seconds = time.time() - epoch_started
        elapsed_seconds += epoch_seconds
        state["local_epochs_completed"] = epoch + 1
        history.append(
            {
                "phase": "local_pretraining",
                "epoch": epoch + 1,
                "batches": int(epoch_batches),
                "epoch_time_seconds": float(epoch_seconds),
                "elapsed_time_seconds": float(elapsed_seconds),
                "fedavg_synchronized": bool(synchronized),
            }
        )
        state["history"] = history
        state["elapsed_seconds_completed"] = float(elapsed_seconds)
        if (epoch + 1) in log_epochs:
            print(
                "[StageB-local] epoch={}/{}".format(epoch + 1, args.local_epochs),
                flush=True,
            )
        if (epoch + 1) % int(args.checkpoint_interval) == 0:
            save_checkpoint(graph, session, args.checkpoint_prefix, state)

    if not state["teacher_initialized"]:
        session.run(graph["teacher_assign"])
        state["teacher_initialized"] = True
        save_checkpoint(graph, session, args.checkpoint_prefix, state)

    log_epochs = set(
        np.linspace(
            1,
            max(1, int(args.distill_epochs)),
            min(10, args.distill_epochs),
        )
        .round()
        .astype(np.int64)
        .tolist()
    )
    for epoch in range(
        int(state["distill_epochs_completed"]), int(args.distill_epochs)
    ):
        epoch_started = time.time()
        epoch_batches = 0
        for client_id, indices in enumerate(client_indices):
            for batch in deterministic_batches(
                indices, args.batch_size, args.seed, 2, epoch, client_id
            ):
                feed = {
                    graph["raw_batch_x"]: data["x_encoder_train"][batch],
                    graph["raw_batch_y"]: one_hot(
                        data["y_encoder_train"][batch], args.num_classes
                    ),
                }
                session.run(graph["distill_ops"][client_id], feed_dict=feed)
                epoch_batches += 1
        epoch_seconds = time.time() - epoch_started
        elapsed_seconds += epoch_seconds
        state["distill_epochs_completed"] = epoch + 1
        history.append(
            {
                "phase": "kd_distillation",
                "epoch": epoch + 1,
                "batches": int(epoch_batches),
                "epoch_time_seconds": float(epoch_seconds),
                "elapsed_time_seconds": float(elapsed_seconds),
            }
        )
        state["history"] = history
        state["elapsed_seconds_completed"] = float(elapsed_seconds)
        if (epoch + 1) in log_epochs:
            print(
                "[StageB-KD] epoch={}/{}".format(epoch + 1, args.distill_epochs),
                flush=True,
            )
        if (epoch + 1) % int(args.checkpoint_interval) == 0:
            save_checkpoint(graph, session, args.checkpoint_prefix, state)

    candidate_fetches = [
        graph["selected_w"].reveal(),
        graph["selected_b"].reveal(),
    ]
    if args.formal_candidate_release:
        candidate_fetches.extend(
            [
                graph["nokd_avg_w"].reveal(),
                graph["nokd_avg_b"].reveal(),
                graph["avg_w"].reveal(),
                graph["avg_b"].reveal(),
                graph["rw_w"].reveal(),
                graph["rw_b"].reveal(),
            ]
        )
    candidate_values = session.run(candidate_fetches)
    selected_w, selected_b = candidate_values[:2]
    released_candidates = None
    diagnostics = None
    revealed_values = ["final_selected_stage_b_weight", "final_selected_stage_b_bias"]
    if args.formal_candidate_release:
        released_candidates = {
            "NoKD": {
                "weight": np.asarray(candidate_values[2], dtype=np.float32),
                "bias": np.asarray(candidate_values[3], dtype=np.float32),
            },
            "KD-Avg": {
                "weight": np.asarray(candidate_values[4], dtype=np.float32),
                "bias": np.asarray(candidate_values[5], dtype=np.float32),
            },
            "KD-ReferenceWeighted": {
                "weight": np.asarray(candidate_values[6], dtype=np.float32),
                "bias": np.asarray(candidate_values[7], dtype=np.float32),
            },
        }
        revealed_values.extend(
            [
                "formal_ablation_nokd_candidate",
                "formal_ablation_kd_avg_candidate",
                "formal_ablation_kd_reference_weighted_candidate",
            ]
        )
    if args.correctness_reveal:
        diagnostic_fetches = [
            graph["teacher_logits"].reveal(),
            graph["avg_w"].reveal(),
            graph["avg_b"].reveal(),
            graph["rw_w"].reveal(),
            graph["rw_b"].reveal(),
            graph["avg_selection_mse"].reveal(),
            graph["rw_selection_mse"].reveal(),
            graph["choose_rw"].reveal(),
        ]
        diagnostic_fetches += [
            value.reveal() for value in graph["normalized_weights"]
        ]
        diagnostic_fetches += [value.reveal() for value in graph["weights"]]
        diagnostic_fetches += [value.reveal() for value in graph["biases"]]
        values = session.run(diagnostic_fetches)
        weight_offset = 8
        local_weight_offset = weight_offset + int(args.num_clients)
        local_bias_offset = local_weight_offset + int(args.num_clients)
        diagnostics = {
            "teacher": np.asarray(values[0]),
            "avg_w": np.asarray(values[1]),
            "avg_b": np.asarray(values[2]),
            "rw_w": np.asarray(values[3]),
            "rw_b": np.asarray(values[4]),
            "avg_selection_mse": float(np.asarray(values[5]).reshape(-1)[0]),
            "rw_selection_mse": float(np.asarray(values[6]).reshape(-1)[0]),
            "choose_rw": int(np.asarray(values[7]).reshape(-1)[0]),
            "reference_weights": np.asarray(
                [
                    float(np.asarray(value).reshape(-1)[0])
                    for value in values[weight_offset:local_weight_offset]
                ]
            ),
            "local_weights": [
                np.asarray(value)
                for value in values[local_weight_offset:local_bias_offset]
            ],
            "local_biases": [
                np.asarray(value)
                for value in values[
                    local_bias_offset : local_bias_offset + int(args.num_clients)
                ]
            ],
        }
        revealed_values.extend(
            [
                "correctness_only_teacher_logits",
                "correctness_only_avg_candidate",
                "correctness_only_reference_weighted_candidate",
                "correctness_only_candidate_losses",
                "correctness_only_selection_bit",
                "correctness_only_reference_weights",
                "correctness_only_local_parameters",
            ]
        )
    save_checkpoint(graph, session, args.checkpoint_prefix, state)
    runtime = time.time() - started
    KE.clear_session()
    return {
        "selected_w": np.asarray(selected_w, dtype=np.float32),
        "selected_b": np.asarray(selected_b, dtype=np.float32),
        "diagnostics": diagnostics,
        "released_candidates": released_candidates,
        "revealed_values": revealed_values,
        "runtime_seconds": float(runtime),
        "training_elapsed_seconds": float(elapsed_seconds),
        "history": history,
        "checkpoint_state": state,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage B：TFE/ABY3 密态 logit-MSE KD 与候选初始化选择"
    )
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument(
        "--feature_cache_mode",
        choices=["npz", "memmap"],
        default="npz",
        help="npz=traditional full NPZ, memmap=indexed access from .npy via metadata NPZ",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument(
        "--split", choices=["iid", "balanced", "mild", "severe"], default="iid"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--local_lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--client_weighting",
        choices=["uniform", "sample_count", "class_balanced"],
        default="uniform",
        help="Public coefficient rule for secure client-head aggregation",
    )
    parser.add_argument(
        "--fedavg_sync_interval",
        type=int,
        default=0,
        help="Synchronize all private client heads every N local epochs; 0 disables",
    )
    parser.add_argument("--distill_epochs", type=int, default=10)
    parser.add_argument("--distill_lr", type=float, default=0.02)
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument(
        "--reference_supervised_weight",
        type=float,
        default=0.0,
        help="Secure calibration loss weight on the independent reference set",
    )
    parser.add_argument("--weight_temperature", type=float, default=0.2)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument(
        "--shared_initialization",
        action="store_true",
        help="Start every client from the same random linear head before local training",
    )
    parser.add_argument("--precision_fractional", type=int, default=16)
    parser.add_argument("--matmul_chunk_size", type=int, default=64)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--correctness_reveal", action="store_true")
    parser.add_argument("--run_plaintext_reference", action="store_true")
    parser.add_argument(
        "--formal_candidate_release",
        action="store_true",
        help=(
            "Release NoKD/KD-Avg/KD-ReferenceWeighted outputs only at the "
            "explicit formal ablation boundary"
        ),
    )
    parser.add_argument(
        "--input_lineage",
        choices=["legacy_unspecified", "official_mtt_only", "full_data_disjoint"],
        default="legacy_unspecified",
    )
    return parser.parse_args()


def validate_input_lineage(args):
    resolved_cache = os.path.realpath(args.feature_cache)
    resolved_output = os.path.realpath(args.output_dir)
    resolved_checkpoint = os.path.realpath(args.checkpoint_prefix)
    forbidden = os.path.join(
        "new_output", "new_incremental_comparison", "stage_b"
    )
    lineage = {
        "initializer_source": args.input_lineage,
        "feature_cache_resolved": resolved_cache,
        "full_data_initializer_dependency": (
            args.input_lineage == "full_data_disjoint"
        ),
        "validated": False,
    }
    if args.input_lineage == "legacy_unspecified":
        return lineage
    if forbidden in resolved_cache:
        raise ValueError("Stage B feature cache resolves into forbidden Full-Data output")
    for value in (resolved_output, resolved_checkpoint):
        if "new_true_ours" not in value:
            raise ValueError("True Ours output/checkpoint must stay under new_true_ours")
    if args.input_lineage == "official_mtt_only":
        required_fragment = os.path.join(
            "new_cache", "new_true_ours", "official_mtt", "stage_b"
        )
        if required_fragment not in resolved_cache or args.feature_cache_mode != "npz":
            raise ValueError("official_mtt_only requires the audited True Ours NPZ")
        with np.load(args.feature_cache, allow_pickle=False) as metadata:
            source = str(metadata["initializer_source"].reshape(-1)[0])
            dependency = int(
                metadata["full_data_initializer_dependency"].reshape(-1)[0]
            )
            split_path = str(metadata["split_manifest_path"].reshape(-1)[0])
            expected_split_sha = str(
                metadata["split_manifest_sha256"].reshape(-1)[0]
            )
        if source != "official_mtt_only" or dependency != 0:
            raise ValueError("Official-MTT lineage metadata failed")
        if not os.path.isfile(split_path) or sha256_file(split_path) != expected_split_sha:
            raise ValueError("Split manifest hash mismatch")
        lineage.update(
            {
                "split_manifest_path": split_path,
                "split_manifest_sha256": expected_split_sha,
                "validated": True,
            }
        )
    else:
        required_fragment = os.path.join(
            "new_cache", "new_true_ours", "full_data_stage_b"
        )
        if required_fragment not in resolved_cache or args.feature_cache_mode != "memmap":
            raise ValueError("full_data_disjoint requires audited memmap metadata")
        lineage["validated"] = True
    return lineage


def main():
    args = parse_args()
    set_seed(args.seed)
    input_lineage = validate_input_lineage(args)
    data = load_feature_cache(args.feature_cache, mode=args.feature_cache_mode)
    args.feature_dim = int(data["feature_dim"])
    args.num_classes = int(
        max(
            np.max(data["y_encoder_train"]),
            np.max(data["y_reference"]),
            np.max(data["y_selection"]),
        )
        + 1
    )
    if args.feature_dim != 2048 and not args.correctness_reveal:
        raise ValueError("正式 Stage B 要求 feature_dim=2048")
    if args.run_plaintext_reference and not args.correctness_reveal:
        raise ValueError("明文参考实现只能用于 correctness，不得进入正式运行")
    client_indices = make_client_splits(
        data["y_encoder_train"], args.num_clients, args.split, args.seed
    )
    args.client_aggregation_weights = aggregation_weights(
        data["y_encoder_train"], client_indices, args.client_weighting
    )
    initial_weights, initial_biases = initialize_heads(
        args.num_clients,
        args.feature_dim,
        args.num_classes,
        args.seed,
        args.init_scale,
    )
    if args.shared_initialization:
        initial_weights = [initial_weights[0].copy() for _ in initial_weights]
        initial_biases = [initial_biases[0].copy() for _ in initial_biases]
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 92, flush=True)
    print("Stage B：TFE/ABY3 Secure Consensus Initialization", flush=True)
    print(
        "dataset={} split={} clients={} feature_dim={} classes={}".format(
            args.dataset,
            args.split,
            args.num_clients,
            args.feature_dim,
            args.num_classes,
        ),
        flush=True,
    )
    print(
        "local_epochs={} distill_epochs={} batch_size={} "
        "cpu_policy=runtime_unrestricted num_workers=N/A".format(
            args.local_epochs, args.distill_epochs, args.batch_size
        ),
        flush=True,
    )
    print(
        "backend=tfe_mpc protocol=ABY3 precision_fractional={} output={}".format(
            args.precision_fractional, args.output_dir
        ),
        flush=True,
    )
    print("=" * 92, flush=True)
    started = time.time()
    secure = run_secure_stage_b(
        data, client_indices, initial_weights, initial_biases, args
    )
    plaintext = None
    correctness = None
    if args.run_plaintext_reference:
        plaintext = plaintext_reference(
            data["x_encoder_train"],
            data["y_encoder_train"],
            data["x_reference"],
            data["y_reference"],
            data["x_selection"],
            data["y_selection"],
            client_indices,
            initial_weights,
            initial_biases,
            args,
        )
        correctness = {
            "selected_weight_max_abs_error": float(
                np.max(np.abs(secure["selected_w"] - plaintext["selected_w"]))
            ),
            "selected_bias_max_abs_error": float(
                np.max(np.abs(secure["selected_b"] - plaintext["selected_b"]))
            ),
            "avg_weight_max_abs_error": float(
                np.max(
                    np.abs(secure["diagnostics"]["avg_w"] - plaintext["avg_w"])
                )
            ),
            "rw_weight_max_abs_error": float(
                np.max(np.abs(secure["diagnostics"]["rw_w"] - plaintext["rw_w"]))
            ),
            "avg_rw_secure_max_abs_difference": float(
                np.max(
                    np.abs(
                        secure["diagnostics"]["avg_w"]
                        - secure["diagnostics"]["rw_w"]
                    )
                )
            ),
            "local_weight_max_abs_errors": [
                float(np.max(np.abs(secure_weight - plaintext_weight)))
                for secure_weight, plaintext_weight in zip(
                    secure["diagnostics"]["local_weights"], plaintext["weights"]
                )
            ],
            "teacher_logits_max_abs_error": float(
                np.max(
                    np.abs(
                        secure["diagnostics"]["teacher"] - plaintext["teacher"]
                    )
                )
            ),
        }
        if secure["released_candidates"] is not None:
            correctness.update(
                {
                    "nokd_avg_weight_max_abs_error": float(
                        np.max(
                            np.abs(
                                secure["released_candidates"]["NoKD"]["weight"]
                                - plaintext["nokd_avg_w"]
                            )
                        )
                    ),
                    "nokd_avg_bias_max_abs_error": float(
                        np.max(
                            np.abs(
                                secure["released_candidates"]["NoKD"]["bias"]
                                - plaintext["nokd_avg_b"]
                            )
                        )
                    ),
                }
            )

    model_path = os.path.join(
        args.output_dir,
        "new_secure_stage_b_selected_{}_{}_seed{}.npz".format(
            args.dataset, args.split, args.seed
        ),
    )
    atomic_npz_save(
        model_path,
        weight=secure["selected_w"],
        bias=secure["selected_b"],
        feature_dim=np.asarray([args.feature_dim], dtype=np.int64),
        num_classes=np.asarray([args.num_classes], dtype=np.int64),
        initializer_source=np.asarray([args.input_lineage]),
        full_data_initializer_dependency=np.asarray(
            [1 if args.input_lineage == "full_data_disjoint" else 0],
            dtype=np.int64,
        ),
        feature_cache_sha256=np.asarray([sha256_file(args.feature_cache)]),
    )
    candidate_paths = {}
    if secure["released_candidates"] is not None:
        for method, candidate in secure["released_candidates"].items():
            safe_name = method.lower().replace("-", "_")
            candidate_path = os.path.join(
                args.output_dir,
                "new_secure_stage_b_candidate_{}_{}_{}_seed{}.npz".format(
                    safe_name, args.dataset, args.split, args.seed
                ),
            )
            atomic_npz_save(
                candidate_path,
                weight=candidate["weight"],
                bias=candidate["bias"],
                feature_dim=np.asarray([args.feature_dim], dtype=np.int64),
                num_classes=np.asarray([args.num_classes], dtype=np.int64),
                initializer_source=np.asarray([args.input_lineage]),
                full_data_initializer_dependency=np.asarray(
                    [1 if args.input_lineage == "full_data_disjoint" else 0],
                    dtype=np.int64,
                ),
                feature_cache_sha256=np.asarray([sha256_file(args.feature_cache)]),
            )
            candidate_paths[method] = {
                "path": candidate_path,
                "sha256": sha256_file(candidate_path),
            }
    report = {
        "schema": "new_secure_stage_b_report_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "split": args.split,
        "feature_cache": os.path.abspath(args.feature_cache),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "initializer_lineage": input_lineage,
        "feature_dim": int(args.feature_dim),
        "num_classes": int(args.num_classes),
        "num_clients": int(args.num_clients),
        "client_splits": split_summary(data["y_encoder_train"], client_indices),
        "configuration": {
            "batch_size": int(args.batch_size),
            "local_epochs": int(args.local_epochs),
            "local_lr": float(args.local_lr),
            "weight_decay": float(args.weight_decay),
            "client_weighting": args.client_weighting,
            "client_aggregation_weights": [
                float(value) for value in args.client_aggregation_weights
            ],
            "fedavg_sync_interval": int(args.fedavg_sync_interval),
            "shared_initialization": bool(args.shared_initialization),
            "distill_epochs": int(args.distill_epochs),
            "distill_lr": float(args.distill_lr),
            "lambda_kd": float(args.lambda_kd),
            "reference_supervised_weight": float(
                args.reference_supervised_weight
            ),
            "weight_temperature": float(args.weight_temperature),
            "precision_fractional": int(args.precision_fractional),
            "matmul_chunk_size": int(args.matmul_chunk_size),
        },
        "security": {
            "backend": "tfe_mpc",
            "protocol": "ABY3",
            "execution": "single-machine process simulation",
            "secret_shared_inputs": [
                "local_feature_batches",
                "local_labels",
                "local_linear_head_parameters",
                "reference_features",
                "reference_labels",
                "selection_features",
                "selection_labels",
            ],
            "secret_shared_intermediates": [
                "teacher_logits",
                "distilled_local_parameters",
                "avg_candidate",
                "reference_weighted_candidate",
                "candidate_mse_scores",
                "selection_bit",
            ],
            "revealed_values": secure["revealed_values"],
            "formal_candidate_release": bool(args.formal_candidate_release),
            "reference_weight_softmax_stabilization": (
                "private_mean_centering_shift_invariant"
            ),
            "selection_scope": (
                "fully_secret_selection_final_model_only"
                if not args.correctness_reveal
                else "correctness_diagnostics_with_explicit_reveals"
            ),
            "silent_plaintext_fallback": False,
            "secure_random_custom_op_required": True,
        },
        "selected_model_path": model_path,
        "selected_model_sha256": sha256_file(model_path),
        "formal_candidate_models": candidate_paths,
        "runtime_seconds": float(time.time() - started),
        "training_elapsed_seconds": secure["training_elapsed_seconds"],
        "history": secure["history"],
        "correctness": correctness,
    }
    if secure["diagnostics"] is not None:
        report["correctness_diagnostics"] = {
            "avg_selection_mse": secure["diagnostics"]["avg_selection_mse"],
            "rw_selection_mse": secure["diagnostics"]["rw_selection_mse"],
            "choose_rw": secure["diagnostics"]["choose_rw"],
            "reference_weights": secure["diagnostics"]["reference_weights"].tolist(),
        }
    report_path = os.path.join(
        args.output_dir,
        "new_secure_stage_b_report_{}_{}_seed{}.json".format(
            args.dataset, args.split, args.seed
        ),
    )
    atomic_json_dump(report, report_path)
    print("selected_model={}".format(model_path), flush=True)
    print("report={}".format(report_path), flush=True)
    if correctness is not None:
        print("correctness={}".format(correctness), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
