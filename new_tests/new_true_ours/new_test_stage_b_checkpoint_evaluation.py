#!/usr/bin/env python3
"""Real TFE/ABY3 test for the formal Stage-B checkpoint trajectory."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_stage_b_secure_init import run_secure_stage_b


def main():
    data = {
        "x_encoder_train": np.asarray(
            [
                [1.0, 0.0, 0.5, -0.5],
                [0.0, 1.0, -0.5, 0.5],
                [0.5, 0.5, 1.0, 0.0],
                [-0.5, 0.5, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "y_encoder_train": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "x_reference": np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "y_reference": np.asarray([0, 1], dtype=np.int64),
    }
    initial_weights = [
        np.asarray(
            [[0.1, -0.1], [0.0, 0.1], [0.05, -0.05], [0.0, 0.0]],
            dtype=np.float32,
        ),
        np.asarray(
            [[0.0, 0.1], [0.1, -0.1], [-0.05, 0.05], [0.0, 0.0]],
            dtype=np.float32,
        ),
    ]
    initial_biases = [
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    ]
    with tempfile.TemporaryDirectory(prefix="new_stage_b_checkpoint_eval_") as temp:
        args = SimpleNamespace(
            batch_size=2,
            feature_dim=4,
            num_classes=2,
            num_clients=2,
            precision_fractional=16,
            matmul_chunk_size=2,
            weight_decay=0.0,
            local_lr=0.05,
            client_aggregation_weights=[0.5, 0.5],
            secure_average_only=True,
            development_checkpoint_evaluation=True,
            fedavg_sync_interval=0,
            local_epochs=1,
            checkpoint_interval=1,
            seed=42,
            checkpoint_prefix=str(Path(temp) / "new_stage_b"),
            distill_epochs=0,
            formal_candidate_release=False,
            correctness_reveal=False,
        )
        result = run_secure_stage_b(
            data,
            [np.asarray([0, 1]), np.asarray([2, 3])],
            initial_weights,
            initial_biases,
            args,
        )
    if len(result["history"]) != 1:
        raise AssertionError("expected exactly one history row")
    row = result["history"][0]
    expected = float(
        np.mean(
            np.argmax(
                np.matmul(data["x_reference"], result["selected_w"])
                + result["selected_b"],
                axis=1,
            )
            == data["y_reference"]
        )
    )
    if row["development_validation_accuracy"] != expected:
        raise AssertionError(
            "secure checkpoint accuracy mismatch: {} vs {}".format(
                row["development_validation_accuracy"], expected
            )
        )
    if (
        row["evaluation_split"] != "development_validation"
        or row["locked_holdout_read"] is not False
        or row["official_test_read"] is not False
    ):
        raise AssertionError("checkpoint evaluation scope metadata failed")
    if (
        "aggregate_development_validation_logits_at_checkpoint_epochs"
        not in result["revealed_values"]
    ):
        raise AssertionError("checkpoint reveal scope was not disclosed")
    print(
        "PASS development_accuracy={} history_rows={}".format(
            expected, len(result["history"])
        )
    )


if __name__ == "__main__":
    main()
