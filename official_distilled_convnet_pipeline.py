#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：跨阶段主流程共享工具。

该模块只实现算法主流程：
official distilled images -> full ConvNet Stage B -> same ConvNet Stage C。
它不提取或缓存 2048 维特征，不调用 TFE，也不把 PyTorch 运算称为 MPC。
"""

import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_paths import PATHS

BASE_DIR = PATHS.output_root
MTT_ROOT = PATHS.mtt_root
OFFICIAL_ROOT = PATHS.official_mtt_root
CHECKPOINT_DIR = BASE_DIR / "init_model" / "cross_stage_convnet"

DATASET_CONFIG = {
    "cifar10": {
        "mtt_dataset": "CIFAR10",
        "model_name": "ConvNetD3",
        "official_dir": OFFICIAL_ROOT / "cifar10_50",
        "data_path": PATHS.cifar10_data_root,
        "num_classes": 10,
        "image_size": (32, 32),
    },
    "cifar100": {
        "mtt_dataset": "CIFAR100",
        "model_name": "ConvNetD3",
        "official_dir": OFFICIAL_ROOT / "cifar100_50",
        "data_path": PATHS.cifar100_data_root,
        "num_classes": 100,
        "image_size": (32, 32),
    },
    "tiny": {
        "mtt_dataset": "Tiny",
        "model_name": "ConvNetD4",
        "official_dir": OFFICIAL_ROOT / "tiny_50",
        "data_path": PATHS.tiny_imagenet_root,
        "num_classes": 200,
        "image_size": (64, 64),
    },
}


def set_safe_runtime(seed: int, cpu_threads: int = 1) -> None:
    """限制线程并固定随机种子；所有数据集和 seed 串行执行。"""
    threads = max(1, int(cpu_threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(threads)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str = "auto") -> torch.device:
    if str(device).lower() != "cpu" and torch.cuda.is_available():
        dev = torch.device("cuda")
        _ = torch.ones(1, device=dev) + 1
        return dev
    return torch.device("cpu")


def _add_mtt_path() -> None:
    if str(MTT_ROOT) not in sys.path:
        sys.path.insert(0, str(MTT_ROOT))


def _reset_module(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def build_convnet(dataset: str, seed: int, device: torch.device) -> nn.Module:
    """构造确定性 MTT ConvNetD3/D4，三个阶段使用同一模型定义。"""
    _add_mtt_path()
    from utils import get_network

    cfg = DATASET_CONFIG[dataset]
    torch.manual_seed(int(seed))
    model = get_network(
        cfg["model_name"],
        channel=3,
        num_classes=int(cfg["num_classes"]),
        im_size=tuple(cfg["image_size"]),
        dist=False,
    )
    # MTT get_network 内部会按时间改随机种子，因此创建后重新确定性初始化。
    torch.manual_seed(int(seed))
    model.apply(_reset_module)
    return model.to(device)


def load_official_distilled_images(dataset: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    folder = DATASET_CONFIG[dataset]["official_dir"]
    image_files = sorted(folder.glob("images_best*.pt"))
    label_files = sorted(folder.glob("labels_best*.pt"))
    if not image_files or len(image_files) != len(label_files):
        raise FileNotFoundError("official distilled files are incomplete: {}".format(folder))
    images, labels = [], []
    for image_path, label_path in zip(image_files, label_files):
        x = torch.load(str(image_path), map_location="cpu").detach().float()
        y = torch.load(str(label_path), map_location="cpu").detach().long().view(-1)
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        images.append(x)
        labels.append(y)
    return torch.cat(images), torch.cat(labels), [p.name for p in image_files]


def _mtt_dataset_objects(dataset: str):
    _add_mtt_path()
    from argparse import Namespace
    from utils import ParamDiffAug, get_daparam, get_dataset

    cfg = DATASET_CONFIG[dataset]
    args = Namespace(
        dataset=cfg["mtt_dataset"],
        model=cfg["model_name"],
        ipc=50,
        device="cpu",
        zca=False,
        res=None,
        subset="imagenette",
        batch_train=64,
        batch_real=64,
        lr_net=0.01,
        epoch_eval_train=0,
        dsa=True,
        dsa_strategy="color_crop_cutout_flip_scale_rotate",
        dsa_param=ParamDiffAug(),
        canvas_size=2,
        canvas_samples=1,
        texture=False,
    )
    args.dc_aug_param = get_daparam(args.dataset, cfg["model_name"], cfg["model_name"], 50)
    return get_dataset(
        cfg["mtt_dataset"],
        str(cfg["data_path"]),
        batch_size=64,
        subset=args.subset,
        args=args,
    )


def _balanced_indices(dataset_obj, num_classes: int, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    by_class = {c: [] for c in range(num_classes)}
    for idx in range(len(dataset_obj)):
        _, label = dataset_obj[idx]
        by_class[int(label)].append(idx)
    selected = []
    for cls in range(num_classes):
        ids = np.asarray(by_class[cls], dtype=np.int64)
        rng.shuffle(ids)
        selected.extend(ids[: min(int(per_class), len(ids))].tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _materialize(dataset_obj, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for idx in indices:
        x, y = dataset_obj[int(idx)]
        xs.append(x.float())
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def load_image_bundle(
    dataset: str,
    reference_per_class: int,
    max_test_samples: int,
    seed: int,
) -> Dict:
    """
    加载蒸馏训练图像、互斥 held-out reference set 和 test set。

    蒸馏图像来自独立 official 文件；reference 来自真实 train split；
    test 来自官方 test/val split，三者通过来源和索引显式隔离。
    """
    x_syn, y_syn, source_files = load_official_distilled_images(dataset)
    channel, im_size, num_classes, _, _, _, train_ds, test_ds, *_ = _mtt_dataset_objects(dataset)
    ref_indices = _balanced_indices(train_ds, num_classes, reference_per_class, seed + 7001)
    x_ref, y_ref = _materialize(train_ds, ref_indices)
    test_count = len(test_ds) if int(max_test_samples) <= 0 else min(len(test_ds), int(max_test_samples))
    test_indices = np.arange(test_count, dtype=np.int64)
    x_test, y_test = _materialize(test_ds, test_indices)
    return {
        "input_type": "nchw_images",
        "feature_extraction_used": False,
        "x_train": x_syn,
        "y_train": y_syn,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "x_test": x_test,
        "y_test": y_test,
        "train_sample_ids": ["official_distilled:{}".format(i) for i in range(len(y_syn))],
        "reference_sample_ids": ["real_train:{}".format(int(i)) for i in ref_indices],
        "test_sample_ids": ["benchmark_test:{}".format(int(i)) for i in test_indices],
        "reference_name": "heldout_reference_set",
        "reference_private_dependency": True,
        "source_files": source_files,
        "channel": int(channel),
        "image_size": tuple(im_size),
        "num_classes": int(num_classes),
        "model_name": DATASET_CONFIG[dataset]["model_name"],
    }


def assert_split_boundaries(bundle: Dict) -> None:
    groups = [
        set(bundle["train_sample_ids"]),
        set(bundle["reference_sample_ids"]),
        set(bundle["test_sample_ids"]),
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert bundle["x_train"].ndim == 4
    assert bundle["x_ref"].ndim == 4
    assert bundle["x_test"].ndim == 4
    assert bundle["feature_extraction_used"] is False


def make_client_splits(y: torch.Tensor, num_clients: int, classes_per_client: Optional[int], seed: int) -> List[np.ndarray]:
    """构造 IID 或 class-skew 划分，返回蒸馏图像索引。"""
    rng = np.random.RandomState(seed)
    labels = y.cpu().numpy().astype(np.int64)
    classes = np.unique(labels)
    if classes_per_client is None or int(classes_per_client) >= len(classes):
        ids = np.arange(len(labels))
        rng.shuffle(ids)
        return [np.asarray(x, dtype=np.int64) for x in np.array_split(ids, num_clients)]
    parts = [[] for _ in range(num_clients)]
    cpc = max(1, int(classes_per_client))
    for class_pos, cls in enumerate(classes):
        ids = np.where(labels == cls)[0]
        rng.shuffle(ids)
        owner_count = max(1, min(num_clients, int(np.ceil(num_clients * cpc / float(len(classes))))))
        owners = [(class_pos + j) % num_clients for j in range(owner_count)]
        for owner, chunk in zip(owners, np.array_split(ids, owner_count)):
            parts[owner].extend(chunk.tolist())
    all_ids = np.arange(len(labels))
    for client_id, ids in enumerate(parts):
        if not ids:
            parts[client_id] = [int(rng.choice(all_ids))]
    return [np.asarray(ids, dtype=np.int64) for ids in parts]


def _sample(x: torch.Tensor, y: torch.Tensor, batch_size: int, rng: np.random.RandomState, device: torch.device):
    ids = rng.choice(len(y), size=min(int(batch_size), len(y)), replace=len(y) < int(batch_size))
    return x[ids].to(device), y[ids].to(device)


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 256) -> Tuple[float, float]:
    original_device = next(model.parameters()).device
    model.to(device).eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(len(yb))
    model.to(original_device)
    return total_loss / max(1, total), correct / max(1, total)


def train_local_convnet(
    dataset: str,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    model = build_convnet(dataset, seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    rng = np.random.RandomState(seed + 19)
    for _ in range(int(steps)):
        xb, yb = _sample(x, y, batch_size, rng, device)
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
    return model.cpu()


def ensemble_teacher_probabilities(models: List[nn.Module], x_ref: torch.Tensor, tau: float, device: torch.device) -> torch.Tensor:
    probabilities = []
    with torch.no_grad():
        for model in models:
            model = model.to(device).eval()
            chunks = []
            for start in range(0, len(x_ref), 128):
                chunks.append(F.softmax(model(x_ref[start:start + 128].to(device)) / float(tau), dim=1).cpu())
            probabilities.append(torch.cat(chunks))
            model.cpu()
    return torch.stack(probabilities).mean(0)


def distill_full_convnet(
    source_model: nn.Module,
    x_local: torch.Tensor,
    y_local: torch.Tensor,
    x_ref: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    tau: float,
    lambda_kd: float,
    device: torch.device,
) -> nn.Module:
    """同时更新 ConvNet feature layers 和 classifier，不冻结任何层。"""
    model = deepcopy(source_model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    rng = np.random.RandomState(seed + 31)
    for _ in range(int(steps)):
        xb, yb = _sample(x_local, y_local, batch_size, rng, device)
        ref_ids = rng.choice(len(x_ref), size=min(int(batch_size), len(x_ref)), replace=False)
        xr = x_ref[ref_ids].to(device)
        teacher = teacher_probabilities[ref_ids].to(device)
        model.train()
        optimizer.zero_grad()
        supervised = F.cross_entropy(model(xb), yb)
        log_prob = F.log_softmax(model(xr) / float(tau), dim=1)
        kd_loss = F.kl_div(log_prob, teacher, reduction="batchmean") * float(tau) ** 2
        loss = supervised + float(lambda_kd) * kd_loss
        loss.backward()
        optimizer.step()
    return model.cpu()


def state_signature(model_or_state) -> str:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_schema(model_or_state) -> Dict[str, Tuple[int, ...]]:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {key: tuple(value.shape) for key, value in state.items()}


def aggregate_full_state(models: List[nn.Module], weights: Optional[Sequence[float]] = None) -> Dict[str, torch.Tensor]:
    if not models:
        raise ValueError("models must not be empty")
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    states = [model.state_dict() for model in models]
    merged = {}
    for key in states[0]:
        first = states[0][key].detach().cpu()
        if not torch.is_floating_point(first):
            merged[key] = first.clone()
            continue
        value = torch.zeros_like(first, dtype=torch.float32)
        for weight, state in zip(weights, states):
            value.add_(state[key].detach().cpu().float(), alpha=float(weight))
        merged[key] = value.to(dtype=first.dtype)
    return merged


def model_from_state(dataset: str, state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    model = build_convnet(dataset, 0, device)
    model.load_state_dict(state, strict=True)
    return model


def build_stage_b_candidates(
    dataset: str,
    distilled_models: List[nn.Module],
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    device: torch.device,
    weight_temperature: float,
) -> Dict:
    ref_scores = []
    for model in distilled_models:
        _, score = evaluate_model(model, x_ref, y_ref, device)
        ref_scores.append(float(score))
    best_index = int(np.argmax(ref_scores))
    z = np.asarray(ref_scores) / max(float(weight_temperature), 1e-6)
    z -= z.max()
    rw_weights = np.exp(z)
    rw_weights /= rw_weights.sum()
    avg = model_from_state(dataset, aggregate_full_state(distilled_models), device).cpu()
    rw = model_from_state(dataset, aggregate_full_state(distilled_models, rw_weights), device).cpu()
    best = deepcopy(distilled_models[best_index]).cpu()
    candidates = {"KD-Avg": avg, "KD-ReferenceWeighted": rw, "KD-Best": best}
    candidate_ref_scores = {
        name: evaluate_model(model, x_ref, y_ref, device)[1]
        for name, model in candidates.items()
    }
    chosen_name = max(candidate_ref_scores, key=candidate_ref_scores.get)
    return {
        "candidates": candidates,
        "chosen_name": chosen_name,
        "chosen_model": deepcopy(candidates[chosen_name]),
        "candidate_ref_scores": candidate_ref_scores,
        "local_ref_scores": ref_scores,
        "reference_weights": rw_weights.tolist(),
        "best_local_index": best_index,
    }


def run_stage_b(
    dataset: str,
    bundle: Dict,
    seed: int,
    num_clients: int,
    classes_per_client: Optional[int],
    local_steps: int,
    kd_steps: int,
    batch_size: int,
    local_lr: float,
    kd_lr: float,
    tau: float,
    lambda_kd: float,
    weight_temperature: float,
    device: torch.device,
) -> Dict:
    assert_split_boundaries(bundle)
    splits = make_client_splits(bundle["y_train"], num_clients, classes_per_client, seed)
    local_models = []
    for client_id, ids in enumerate(splits):
        local_models.append(train_local_convnet(
            dataset,
            bundle["x_train"][ids],
            bundle["y_train"][ids],
            seed + client_id,
            local_steps,
            batch_size,
            local_lr,
            device,
        ))
    teacher = ensemble_teacher_probabilities(local_models, bundle["x_ref"], tau, device)
    distilled_models = []
    before_signatures, after_signatures = [], []
    for client_id, (model, ids) in enumerate(zip(local_models, splits)):
        before_signatures.append(state_signature(model))
        updated = distill_full_convnet(
            model,
            bundle["x_train"][ids],
            bundle["y_train"][ids],
            bundle["x_ref"],
            teacher,
            seed + 100 + client_id,
            kd_steps,
            batch_size,
            kd_lr,
            tau,
            lambda_kd,
            device,
        )
        after_signatures.append(state_signature(updated))
        distilled_models.append(updated)
    candidate_output = build_stage_b_candidates(
        dataset, distilled_models, bundle["x_ref"], bundle["y_ref"], device, weight_temperature
    )
    return {
        "backend": "local_plaintext_pytorch",
        "input_type": bundle["input_type"],
        "input_shape": tuple(bundle["x_train"].shape),
        "feature_extraction_used": False,
        "model_class": type(candidate_output["chosen_model"]).__name__,
        "model_name": bundle["model_name"],
        "model_schema": state_schema(candidate_output["chosen_model"]),
        "chosen_name": candidate_output["chosen_name"],
        "chosen_model": candidate_output["chosen_model"],
        "chosen_signature": state_signature(candidate_output["chosen_model"]),
        "candidates": candidate_output["candidates"],
        "candidate_ref_scores": candidate_output["candidate_ref_scores"],
        "reference_weights": candidate_output["reference_weights"],
        "client_indices": [ids.tolist() for ids in splits],
        "kd_parameter_changed": [a != b for a, b in zip(before_signatures, after_signatures)],
        "local_models": local_models,
        "distilled_models": distilled_models,
    }


def save_stage_b_checkpoint(dataset: str, stage_b: Dict, seed: int, path: Optional[Path] = None) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = CHECKPOINT_DIR / "crossStageConvNet_stageB_chosen_{}_seed{}.pt".format(dataset, seed)
    payload = {
        "schema_version": 1,
        "pipeline": "algorithmic_pytorch_convnet",
        "dataset": dataset,
        "seed": int(seed),
        "input_type": stage_b["input_type"],
        "feature_extraction_used": False,
        "model_name": stage_b["model_name"],
        "model_class": stage_b["model_class"],
        "model_schema": stage_b["model_schema"],
        "chosen_name": stage_b["chosen_name"],
        "heldout_reference_scores": stage_b["candidate_ref_scores"],
        "state_signature": stage_b["chosen_signature"],
        "state_dict": {k: v.detach().cpu() for k, v in stage_b["chosen_model"].state_dict().items()},
    }
    torch.save(payload, str(path))
    return path


def load_stage_b_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("pipeline") != "algorithmic_pytorch_convnet":
        raise RuntimeError("checkpoint is not a full-ConvNet cross-stage checkpoint")
    if payload.get("feature_extraction_used") is not False:
        raise RuntimeError("feature pipeline checkpoint is not accepted")
    model = model_from_state(payload["dataset"], payload["state_dict"], device)
    actual = state_signature(model)
    if actual != payload["state_signature"]:
        raise RuntimeError("checkpoint state signature mismatch")
    payload["checkpoint_sha256"] = file_sha256(path)
    return model, payload


def _per_example_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Tuple[List[List[torch.Tensor]], List[float]]:
    """受控逐样本梯度；仅处理当前小 batch，避免展开巨型中间张量。"""
    gradients, norms = [], []
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    for index in range(len(yb)):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb[index:index + 1]), yb[index:index + 1])
        loss.backward()
        sample_grad = [
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ]
        norm_sq = sum(float(torch.sum(grad * grad).item()) for grad in sample_grad)
        gradients.append(sample_grad)
        norms.append(float(np.sqrt(max(norm_sq, 1e-24))))
    return gradients, norms


def train_stage_c_convnet(
    model: nn.Module,
    bundle: Dict,
    mode: str,
    steps: int,
    batch_size: int,
    lr: float,
    clip_c: float,
    noise_multiplier: float,
    gamma: float,
    eta_c: float,
    seed: int,
    device: torch.device,
    eval_interval: int,
    target_reference_accuracy: Optional[float] = None,
) -> Dict:
    """在同一完整 ConvNet 上执行 Plain、Fixed-DP 或 Adaptive-DP。"""
    if mode not in ("plain", "fixed_dp", "adaptive_dp"):
        raise ValueError("unsupported mode: {}".format(mode))
    model = deepcopy(model).to(device)
    initialization_signature = state_signature(model)
    rng = np.random.RandomState(seed + 404)
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    current_c = float(clip_c)
    history = []
    started = time.time()
    target_step = None
    for step in range(1, int(steps) + 1):
        xb, yb = _sample(bundle["x_train"], bundle["y_train"], batch_size, rng, device)
        c_used = current_c
        if mode == "plain":
            model.train()
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            mean_norm = float(np.sqrt(sum(float(torch.sum(p.grad * p.grad).item()) for p in parameters if p.grad is not None)))
            layer_norms = {
                name: float(torch.linalg.vector_norm(parameter.grad).item())
                if parameter.grad is not None else 0.0
                for name, parameter in named_parameters
            }
            clipped_fraction = 0.0
            with torch.no_grad():
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-float(lr))
        else:
            model.train()
            sample_gradients, norms = _per_example_gradients(model, xb, yb)
            clipped_count = 0
            sums = [torch.zeros_like(parameter) for parameter in parameters]
            for sample_grads, norm in zip(sample_gradients, norms):
                factor = min(1.0, c_used / max(float(norm), 1e-12))
                clipped_count += int(factor < 1.0)
                for accumulator, grad in zip(sums, sample_grads):
                    accumulator.add_(grad, alpha=factor)
            with torch.no_grad():
                for parameter, accumulator in zip(parameters, sums):
                    update = accumulator / float(len(yb))
                    if float(noise_multiplier) > 0:
                        update.add_(torch.randn_like(update), alpha=float(noise_multiplier) * c_used / float(len(yb)))
                    parameter.add_(update, alpha=-float(lr))
            mean_norm = float(np.mean(norms))
            layer_norms = {
                name: float(np.mean([
                    torch.linalg.vector_norm(sample_grads[index]).item()
                    for sample_grads in sample_gradients
                ]))
                for index, (name, _) in enumerate(named_parameters)
            }
            clipped_fraction = float(clipped_count / max(1, len(norms)))
            if mode == "adaptive_dp":
                # 本轮使用 C_t，更新后的 C_{t+1} 从下一轮开始使用。
                current_c = float(np.clip(
                    current_c * np.exp(-float(eta_c) * (float(np.mean(np.asarray(norms) <= current_c)) - float(gamma))),
                    1e-3,
                    100.0,
                ))
        if step % max(1, int(eval_interval)) == 0 or step == int(steps):
            ref_loss, ref_acc = evaluate_model(model, bundle["x_ref"], bundle["y_ref"], device)
            test_loss, test_acc = evaluate_model(model, bundle["x_test"], bundle["y_test"], device)
            if target_reference_accuracy is not None and target_step is None and ref_acc >= target_reference_accuracy:
                target_step = int(step)
            history.append({
                "step": int(step),
                "reference_loss": float(ref_loss),
                "reference_accuracy": float(ref_acc),
                "test_loss_readonly": float(test_loss),
                "test_accuracy_readonly": float(test_acc),
                "mean_total_grad_norm": float(mean_norm),
                "mean_layer_grad_norms": layer_norms,
                "clipped_fraction": float(clipped_fraction),
                "clipping_bound_t": float(c_used),
                "clipping_bound_next": float(current_c),
            })
    return {
        "pipeline": "algorithmic_pytorch_convnet",
        "backend": "local_plaintext_pytorch",
        "mode": mode,
        "privacy_scope": "none" if mode == "plain" else "stage_c_only",
        "feature_extraction_used": False,
        "model": model.cpu(),
        "initialization_signature": initialization_signature,
        "final_signature": state_signature(model),
        "history": history,
        "target_reference_accuracy": target_reference_accuracy,
        "step_at_target_reference_accuracy": target_step,
        "time_s": float(time.time() - started),
        "final_test_accuracy": float(history[-1]["test_accuracy_readonly"]),
        "final_reference_accuracy": float(history[-1]["reference_accuracy"]),
        "final_c": float(history[-1]["clipping_bound_next"]),
    }


def json_safe_stage_b(stage_b: Dict) -> Dict:
    return {
        "backend": stage_b["backend"],
        "input_type": stage_b["input_type"],
        "input_shape": stage_b["input_shape"],
        "feature_extraction_used": stage_b["feature_extraction_used"],
        "model_class": stage_b["model_class"],
        "model_name": stage_b["model_name"],
        "model_schema": {k: list(v) for k, v in stage_b["model_schema"].items()},
        "chosen_name": stage_b["chosen_name"],
        "chosen_signature": stage_b["chosen_signature"],
        "candidate_ref_scores": stage_b["candidate_ref_scores"],
        "reference_weights": stage_b["reference_weights"],
        "kd_parameter_changed": stage_b["kd_parameter_changed"],
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
