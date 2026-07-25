#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a frozen DP-MTT encoder on an external public MTT reference set.

This entry point never loads the benchmark test set or raw private training
records.  It is used only for Stage-A validation/tuning after DP trajectories
have already been released.  Hyperparameters are therefore selected with a
public synthetic reference rather than repeatedly querying test accuracy.
"""

import argparse
import json
from pathlib import Path

import torch

from new_secure_feature_models import build_mtt_convnet
from new_secure_feature_models import configure_single_thread_runtime
from new_secure_feature_models import sha256_file
from new_stage_a_shared_encoder import evaluate_tensor
from new_stage_a_shared_encoder import load_public_mtt


def atomic_json_dump(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen Stage-A encoder on public MTT reference data"
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "tiny"), required=True
    )
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    configure_single_thread_runtime(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no compatible GPU is available")
    if not args.encoder.exists():
        raise FileNotFoundError(args.encoder)

    checkpoint = torch.load(
        str(args.encoder), map_location=device, weights_only=False
    )
    model = build_mtt_convnet(args.dataset, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    public_reference = load_public_mtt(args.dataset)
    images = public_reference["images"]
    labels = public_reference["labels"]
    accuracy = evaluate_tensor(
        model, images, labels, device, int(args.batch_size)
    )

    payload = {
        "schema": "new_stage_a_public_reference_evaluation_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "encoder": str(args.encoder.resolve()),
        "encoder_sha256": sha256_file(args.encoder),
        "reference_source": "official_mtt_ipc50_public_synthetic",
        "reference_samples": int(len(labels)),
        "public_reference_accuracy": float(accuracy),
        "test_loaded": False,
        "raw_private_data_loaded": False,
        "selection_only": True,
        "cpu_threads": 1,
        "num_workers": 0,
    }
    atomic_json_dump(payload, args.output_json)
    print("=" * 84)
    print("Stage-A external public reference evaluation")
    print(
        "dataset={} samples={} accuracy={:.6f}".format(
            args.dataset, len(labels), accuracy
        )
    )
    print("test_loaded=False raw_private_data_loaded=False")
    print("output={}".format(args.output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
