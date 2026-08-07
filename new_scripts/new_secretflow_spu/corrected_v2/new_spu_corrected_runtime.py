#!/usr/bin/env python3
"""Security/runtime primitives for the corrected-v2 SPU pipeline.

The predecessor formal runner revealed client-local Stage-B models and created
all deterministic DP-noise shares on the driver.  This module provides the
opposite invariants:

* multi-output SPU functions stay as SPUObject values;
* synchronization uses ``sf.wait`` without public reconstruction;
* every explicit public reconstruction is recorded by a reveal ledger;
* Gaussian shares are sampled independently inside PYU workers from OS CSPRNG
  entropy and are transferred directly to SPU;
* SPU share checkpoints use per-party ``spu.dump`` files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy


def spu_multi_return(spu_device, function):
    """Compile a function while retaining each output as an SPUObject."""
    return spu_device(
        function,
        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_COMPILER,
    )


class RevealLedger:
    """Single explicit boundary for public reconstruction in corrected code."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def reveal(self, label: str, value: Any, scope: str, reason: str) -> Any:
        if not label or not scope or not reason:
            raise ValueError("reveal label/scope/reason must be non-empty")
        result = sf.reveal(value)
        self.rows.append(
            {
                "label": label,
                "scope": scope,
                "reason": reason,
                "public_reconstruction": True,
            }
        )
        return result

    def record_wait(self, label: str, scope: str) -> None:
        self.rows.append(
            {
                "label": label,
                "scope": scope,
                "reason": "secure synchronization; sf.wait returns no value",
                "public_reconstruction": False,
            }
        )

    def wait(self, label: str, value: Any, scope: str) -> None:
        sf.wait(value)
        self.record_wait(label, scope)


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _normal_from_os_csprng(shape: Sequence[int], stddev: float) -> np.ndarray:
    """Box-Muller Gaussian samples whose uniforms come from ``os.urandom``."""
    shape = tuple(int(v) for v in shape)
    count = int(np.prod(shape, dtype=np.int64))
    pair_count = (count + 1) // 2
    raw = os.urandom(pair_count * 16)
    words = np.frombuffer(raw, dtype="<u8")
    # Map to the open interval (0, 1); 53 bits match float64 precision.
    uniforms = ((words >> 11).astype(np.float64) + 0.5) / float(1 << 53)
    u1 = np.maximum(uniforms[0::2], np.finfo(np.float64).tiny)
    u2 = uniforms[1::2]
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * math.pi * u2
    values = np.empty(pair_count * 2, dtype=np.float64)
    values[0::2] = radius * np.cos(angle)
    values[1::2] = radius * np.sin(angle)
    return (values[:count] * float(stddev)).reshape(shape).astype(np.float32)


def private_csprng_gaussian_share(
    state_path: str,
    shape: Sequence[int],
    stddev: float,
    stream: str,
) -> np.ndarray:
    """PYU worker function: generate one private Gaussian share.

    Only the returned array is moved directly from its PYU into SPU.  The
    driver must never call ``sf.reveal`` on that PYUObject.  Persistent state
    contains a monotonic request counter and a one-way share commitment, never
    entropy, a seed, or the share itself.  Fresh OS entropy on every request
    means a crash/resume cannot repeat DP noise.
    """
    path = Path(state_path)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        counter = int(state["requests_completed"])
    else:
        counter = 0
    share = _normal_from_os_csprng(shape, stddev)
    commitment = hashlib.sha256(
        stream.encode("utf-8")
        + b"\0"
        + str(counter).encode("ascii")
        + b"\0"
        + share.tobytes(order="C")
    ).hexdigest()
    _atomic_private_json(
        path,
        {
            "schema": "new_spu_private_noise_state_v1",
            "stream": stream,
            "requests_completed": counter + 1,
            "last_share_commitment": commitment,
            "entropy_persisted": False,
            "share_persisted": False,
        },
    )
    return share


def private_noise_state_public(state_path: str) -> dict[str, Any]:
    """PYU worker function returning non-secret resume/audit metadata only."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return {
        "schema": state["schema"],
        "stream": state["stream"],
        "requests_completed": int(state["requests_completed"]),
        "last_share_commitment": state["last_share_commitment"],
        "entropy_persisted": bool(state["entropy_persisted"]),
        "share_persisted": bool(state["share_persisted"]),
    }


def request_private_noise_shares(
    parties: Iterable[Any],
    spu_device: Any,
    state_root: Path,
    run_id: str,
    stream: str,
    shape: Sequence[int],
    stddev: float = 1.0 / math.sqrt(3.0),
) -> tuple[list[Any], list[str]]:
    """Request one independently sampled share from each PYU.

    The driver receives only PYUObject handles.  Each handle is transferred
    directly to SPU; no share is reconstructed on the driver.
    """
    spu_shares = []
    state_paths = []
    for party in parties:
        party_name = str(party.party)
        state_path = state_root / run_id / party_name / "{}.json".format(stream)
        state_paths.append(str(state_path))
        private_object = party(private_csprng_gaussian_share)(
            str(state_path), tuple(int(v) for v in shape), float(stddev), stream
        )
        spu_shares.append(private_object.to(spu_device))
    return spu_shares, state_paths


def reveal_private_noise_metadata(
    parties: Sequence[Any],
    state_paths: Sequence[str],
    ledger: RevealLedger,
    stream: str,
) -> list[dict[str, Any]]:
    """Reveal counters/commitments only; never reveal seeds or shares."""
    objects = [
        party(private_noise_state_public)(path)
        for party, path in zip(parties, state_paths)
    ]
    rows = ledger.reveal(
        "{}_provider_metadata".format(stream),
        objects,
        "provider_public_metadata",
        "audit monotonic counters and distinct commitments without noise reveal",
    )
    commitments = [row["last_share_commitment"] for row in rows]
    if len(set(commitments)) != len(commitments):
        raise AssertionError("party noise commitments are not distinct")
    if any(row["entropy_persisted"] or row["share_persisted"] for row in rows):
        raise AssertionError("private provider persisted entropy or a noise share")
    return rows


def secure_share_checkpoint_paths(
    checkpoint_root: Path,
    run_id: str,
    object_name: str,
    parties: Sequence[str] = ("alice", "bob", "carol"),
) -> list[str]:
    return [
        str(checkpoint_root / run_id / "spu_shares" / party / "{}.pkl".format(object_name))
        for party in parties
    ]


def dump_spu_object(
    spu_device: Any,
    value: Any,
    checkpoint_root: Path,
    run_id: str,
    object_name: str,
) -> list[str]:
    paths = secure_share_checkpoint_paths(checkpoint_root, run_id, object_name)
    spu_device.dump(value, paths)
    return paths


def load_spu_object(spu_device: Any, paths: Sequence[str]) -> Any:
    return spu_device.load(list(paths))
