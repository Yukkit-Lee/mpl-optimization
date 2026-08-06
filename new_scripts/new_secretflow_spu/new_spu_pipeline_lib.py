#!/usr/bin/env python3
"""Shared data-loading, splitting, hashing, and plaintext-reference helpers
for the isolated SecretFlow-SPU paper pipeline.

Everything here mirrors the audited TFE semantics (see new_stage_b_secure_init.py
and new_stage_c_secure_dpsgd.py) without importing or editing any TFE code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return result


def softmax_plain(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float32)
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def mse_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    error = logits - targets
    return float(np.mean(error * error))


def require_finite_in_chunks(value: np.ndarray, chunk: int = 4096) -> None:
    value = np.asarray(value)
    for offset in range(0, value.shape[0], chunk):
        block = value[offset : offset + chunk]
        if not np.all(np.isfinite(block.astype(np.float64))):
            raise ValueError("non-finite values detected")


class MemmapIndexedArray:
    """Small lazy memmap-backed array compatible with the TFE Full path."""

    def __init__(self, feature_npy_path: str, indices: np.ndarray):
        self._mmap = np.load(feature_npy_path, mmap_mode="r")
        self._indices = np.asarray(indices, dtype=np.int64)
        self._n = len(self._indices)

    @property
    def shape(self):
        return (self._n,) + self._mmap.shape[1:]

    @property
    def dtype(self):
        return self._mmap.dtype

    def __len__(self):
        return self._n

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self._n
            batch_indices = self._indices[start:stop]
            return np.array(self._mmap[batch_indices], dtype=np.float32)
        elif isinstance(key, int):
            return np.array(self._mmap[self._indices[key]], dtype=np.float32)
        batch_indices = self._indices[key]
        return np.array(self._mmap[batch_indices], dtype=np.float32)


def load_ours_cache(cache_path: str) -> Dict[str, Any]:
    """Load an Official-MTT compact cache (npz, full arrays in memory)."""
    with np.load(cache_path, allow_pickle=False) as cache:
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
            raise ValueError("feature cache missing: {}".format(missing))
        result = {name: np.asarray(cache[name]) for name in required}
    result["x_encoder_train"] = result["x_encoder_train"].astype(np.float32)
    result["x_reference"] = result["x_reference"].astype(np.float32)
    result["x_selection"] = result["x_selection"].astype(np.float32)
    result["y_encoder_train"] = result["y_encoder_train"].astype(np.int64)
    result["y_reference"] = result["y_reference"].astype(np.int64)
    result["y_selection"] = result["y_selection"].astype(np.int64)
    result["feature_dim"] = int(result["feature_dim"].reshape(-1)[0])
    return result


def load_full_cache(metadata_path: str) -> Dict[str, Any]:
    """Load a Full-data cache backed by the memmap feature store."""
    meta = np.load(metadata_path, allow_pickle=False)
    feature_npy_path = str(
        np.asarray(meta["feature_npy_path"]).reshape(-1)[0], encoding="utf-8"
    )
    train_indices = np.asarray(meta["train_indices"], dtype=np.int64)
    result = {
        "x_encoder_train": MemmapIndexedArray(feature_npy_path, train_indices),
        "y_encoder_train": np.asarray(meta["y_encoder_train"], dtype=np.int64),
        "x_reference": np.asarray(meta["x_reference"], dtype=np.float32),
        "y_reference": np.asarray(meta["y_reference"], dtype=np.int64),
        "x_selection": np.asarray(meta["x_selection"], dtype=np.float32),
        "y_selection": np.asarray(meta["y_selection"], dtype=np.int64),
        "feature_dim": 2048,
    }
    return result


def make_client_splits(labels: np.ndarray, num_clients: int, split: str, seed: int):
    """Replicate the audited balanced client partition from TFE."""
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
            class_rng = np.random.RandomState(int(seed) + int(class_id) * 1009 + 728)
            class_indices = class_rng.permutation(class_indices)
            for client_id, part in enumerate(np.array_split(class_indices, num_clients)):
                buckets[client_id].extend(part.tolist())
        return [
            np.random.RandomState(int(seed) + client_id * 7919).permutation(
                np.asarray(bucket, dtype=np.int64)
            )
            for client_id, bucket in enumerate(buckets)
        ]
    raise ValueError("unsupported split kind: {}".format(split))


def aggregation_weights(labels: np.ndarray, client_indices: List[np.ndarray], mode: str):
    """Public aggregation coefficients from split metadata only."""
    if mode == "uniform":
        values = np.ones(len(client_indices), dtype=np.float64)
    elif mode == "sample_count":
        values = np.asarray([len(indices) for indices in client_indices], dtype=np.float64)
    else:
        raise ValueError("unknown client weighting: {}".format(mode))
    return values / np.sum(values)


def deterministic_batches(indices, batch_size, seed, phase, epoch, client_id):
    """Fixed-order full batches with tail cyclic fill to keep static shape."""
    indices = np.asarray(indices, dtype=np.int64)
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


def mtt_anchor_ids(labels: np.ndarray, classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic per-class 40 local + 8 anchor split (Official-MTT)."""
    local_parts = []
    anchor_parts = []
    for class_id in range(classes):
        ids = np.flatnonzero(labels == class_id)
        shuffled = np.random.RandomState(
            42 * 1000003 + class_id * 1009 + 729
        ).permutation(ids)
        anchor_parts.append(shuffled[:8])
        local_parts.append(shuffled[8:])
    return (
        np.sort(np.concatenate(local_parts).astype(np.int64)),
        np.sort(np.concatenate(anchor_parts).astype(np.int64)),
    )


def poisson_batch_indices(num_samples: int, expected_batch_size: int, random_bytes=None):
    """CSPRNG Poisson sampling (host only produces sampling IDs)."""
    if random_bytes is None:
        random_bytes = os.urandom
    sample_rate = float(expected_batch_size) / float(num_samples)
    if sample_rate >= 1.0:
        return np.arange(num_samples, dtype=np.int64)
    threshold = int(sample_rate * float(1 << 64))
    draws = np.frombuffer(random_bytes(num_samples * 8), dtype="<u8", count=num_samples)
    return np.flatnonzero(draws < threshold).astype(np.int64)
