#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Memmap adapter for loading Full Data features in Stage B.

This module provides a drop-in replacement for the NPZ-based `load_feature_cache`
in new_stage_b_secure_init.py. It reads a small metadata NPZ (containing indices
and reference/selection arrays) and maps x_encoder_train to the existing .npy
feature files via memmap.

Compatible with Python 3.6+ (legacy TFE environment).

Usage in Stage B:
    from new_full_feature_memmap_loader import load_feature_cache_memmap
    data = load_feature_cache_memmap(metadata_npz_path)
    # data["x_encoder_train"] is a MemmapIndexedArray that supports slicing
"""

import numpy as np
import os


class MemmapIndexedArray:
    """A numpy-array-like object backed by a memmapped .npy file.

    Supports slice-based indexing (start:stop) by mapping through an index array.
    For small slices, reads from mmap on demand. For large slices, creates a copy.
    """

    def __init__(self, feature_npy_path, indices):
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
        else:
            # Fallback: numpy-style indexing
            batch_indices = self._indices[key]
            return np.array(self._mmap[batch_indices], dtype=np.float32)


def load_feature_cache_memmap(metadata_path):
    """Load feature cache from metadata NPZ with memmap backing.

    Args:
        metadata_path: Path to the metadata NPZ file produced by
            new_prepare_incremental_full_feature_assets.py.

    Returns:
        dict compatible with Stage B's existing interface:
        - x_encoder_train: MemmapIndexedArray (lazy, supports slicing)
        - y_encoder_train: np.ndarray (int64)
        - x_reference: np.ndarray (float32)
        - y_reference: np.ndarray (int64)
        - x_selection: np.ndarray (float32)
        - y_selection: np.ndarray (int64)
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("metadata NPZ not found: {}".format(metadata_path))

    meta = np.load(metadata_path, allow_pickle=False)

    # Extract feature .npy path from metadata (encoded as bytes array)
    feature_npy_bytes = meta.get("feature_npy_path")
    if feature_npy_bytes is not None:
        feature_npy_path = str(np.asarray(feature_npy_bytes).reshape(-1)[0],
                               encoding="utf-8")
    else:
        raise KeyError("metadata NPZ missing 'feature_npy_path'")

    train_indices = np.asarray(meta["train_indices"], dtype=np.int64)
    y_train = np.asarray(meta["y_encoder_train"], dtype=np.int64)
    x_ref = np.asarray(meta["x_reference"], dtype=np.float32)
    y_ref = np.asarray(meta["y_reference"], dtype=np.int64)
    x_sel = np.asarray(meta["x_selection"], dtype=np.float32)
    y_sel = np.asarray(meta["y_selection"], dtype=np.int64)

    x_train = MemmapIndexedArray(feature_npy_path, train_indices)

    return {
        "x_encoder_train": x_train,
        "y_encoder_train": y_train,
        "x_reference": x_ref,
        "y_reference": y_ref,
        "x_selection": x_sel,
        "y_selection": y_sel,
        "feature_dim": 2048,
    }
