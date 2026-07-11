#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 b（KD-based Heterogeneity-aware Initialization）最简可运行实现。

说明：
1) 本脚本是“特征级”框架，不依赖原始图像/文本，直接接收模块 a 输出的特征：
   X_train_feat, y_train, X_test_feat, y_test
2) 兼容二分类/多分类。
3) 核心流程：
   non-IID 划分 -> local 训练 -> teacher ensemble -> KD 蒸馏 -> avg/best 候选初始化 -> 选初始全局模型
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
INIT_MODEL_DIR = BASE_DIR / "init_model"


def set_seed(seed: int = 42) -> None:
    """固定随机种子，提升可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_log_epochs(total_epochs: int, max_lines: int = 10) -> set:
    """构造日志采样 epoch，保证输出不超过 max_lines 并覆盖首尾。"""
    total_epochs = max(1, int(total_epochs))
    max_lines = max(1, int(max_lines))
    if total_epochs <= max_lines:
        return set(range(1, total_epochs + 1))
    picks = np.linspace(1, total_epochs, num=max_lines, dtype=int)
    out = set(int(x) for x in picks.tolist())
    out.add(1)
    out.add(total_epochs)
    return out


def infer_num_classes(y: np.ndarray) -> int:
    """根据标签推断类别数。"""
    y = np.asarray(y).reshape(-1)
    return int(len(np.unique(y)))


class LinearClassifier(nn.Module):
    """最简线性分类器：Linear(feat_dim, out_dim)。"""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        out_dim = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_model(feat_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = LinearClassifier(feat_dim, num_classes)
    return model.to(device)


def to_tensor_x(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def to_tensor_y(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    y = np.asarray(y).reshape(-1)
    if num_classes == 2:
        return torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)
    return torch.from_numpy(y.astype(np.int64)).to(device)


def logits_to_probs2(logits: torch.Tensor, num_classes: int, tau: float = 1.0) -> torch.Tensor:
    """将模型输出统一转成概率分布（二分类返回 [N,2]）。"""
    tau = max(float(tau), 1e-6)
    if num_classes == 2:
        p1 = torch.sigmoid(logits / tau)
        p0 = 1.0 - p1
        return torch.cat([p0, p1], dim=1)
    return F.softmax(logits / tau, dim=1)


def predict_labels(model: nn.Module, x: np.ndarray, num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor_x(x, device))
        if num_classes == 2:
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)
    return pred.detach().cpu().numpy()


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray,
                   num_classes: int, device: torch.device) -> float:
    pred = predict_labels(model, x, num_classes, device)
    return accuracy_np(y, pred)


def _ensure_init_model_dir() -> None:
    """确保模块 b 的初始化模型缓存目录存在。"""
    INIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """将模型权重转成 CPU tensor，便于跨设备加载。"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _save_module_b_init_cache(
    dataset_name: str,
    feat_dim: int,
    num_classes: int,
    candidate: Dict,
    config: Dict,
    chosen_model: nn.Module,
    avg_model: nn.Module,
    best_model: nn.Module,
    x_train_shape: Tuple[int, ...],
    x_test_shape: Tuple[int, ...],
    y_train_shape: Tuple[int, ...],
    y_test_shape: Tuple[int, ...],
) -> Dict:
    """
    保存模块 b 当次运行产出的初始化模型到 init_model 目录。
    说明：仅保留 latest 语义，不再维护跨运行“历史最佳”缓存。
    """
    _ensure_init_model_dir()

    dataset = str(dataset_name).strip() or "unknown"
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    cur_score = float(candidate.get("chosen_test_acc", 0.0))

    latest_pt = INIT_MODEL_DIR / f"module_b_init_{dataset}.pt"
    latest_json = INIT_MODEL_DIR / f"module_b_init_{dataset}.json"

    payload = {
        "saved_at": run_ts,
        "dataset": dataset,
        "feat_dim": int(feat_dim),
        "num_classes": int(num_classes),
        "candidate": candidate,
        "config": config,
        "signature": {
            "x_train_shape": tuple(x_train_shape),
            "x_test_shape": tuple(x_test_shape),
            "y_train_shape": tuple(y_train_shape),
            "y_test_shape": tuple(y_test_shape),
        },
        "state_dicts": {
            "initial_global_model": _state_dict_to_cpu(chosen_model),
            "avg_model": _state_dict_to_cpu(avg_model),
            "best_model": _state_dict_to_cpu(best_model),
        },
    }
    torch.save(payload, str(latest_pt))

    latest_meta = {
        "saved_at": run_ts,
        "dataset": dataset,
        "chosen_test_acc": cur_score,
        "chosen_init_type": candidate.get("chosen_init_type", "unknown"),
        "secure_aggregation": bool(candidate.get("secure_aggregation", False)),
        "latest_pt": str(latest_pt),
    }
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(latest_meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "latest_pt": str(latest_pt),
        "latest_json": str(latest_json),
        "score": cur_score,
    }


def make_class_skew_splits(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    classes_per_client: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """构造最简 class-skew non-IID 划分。"""
    rng = np.random.RandomState(seed)

    X = np.asarray(X_train_feat, dtype=np.float32)
    y = np.asarray(y_train).reshape(-1)
    classes = np.unique(y)
    n_classes = len(classes)

    if classes_per_client is None:
        classes_per_client = max(1, n_classes // 2)
    classes_per_client = int(max(1, min(classes_per_client, n_classes)))

    classes_shuffled = classes.copy()
    rng.shuffle(classes_shuffled)
    client_class_sets: List[set] = []
    for cid in range(num_clients):
        cls_set = set(classes_shuffled[(cid + j) % n_classes] for j in range(classes_per_client))
        client_class_sets.append(cls_set)

    class_to_clients: Dict[int, List[int]] = {}
    for c in classes:
        holders = [cid for cid, s in enumerate(client_class_sets) if c in s]
        if not holders:
            cid = int(rng.randint(0, num_clients))
            client_class_sets[cid].add(c)
            holders = [cid]
        class_to_clients[int(c)] = holders

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        holders = class_to_clients[int(c)]
        for k, sample_idx in enumerate(idx):
            target_cid = holders[k % len(holders)]
            client_indices[target_cid].append(int(sample_idx))

    for cid in range(num_clients):
        if len(client_indices[cid]) == 0:
            donor = int(np.argmax([len(v) for v in client_indices]))
            take = max(1, len(client_indices[donor]) // 10)
            moved = client_indices[donor][:take]
            client_indices[donor] = client_indices[donor][take:]
            client_indices[cid].extend(moved)

    clients: List[Dict] = []
    for cid in range(num_clients):
        idx = np.array(client_indices[cid], dtype=np.int64)
        rng.shuffle(idx)
        Xi, yi = X[idx], y[idx]
        labels, counts = np.unique(yi, return_counts=True)
        class_hist = {int(k): int(v) for k, v in zip(labels, counts)}
        clients.append({
            "client_id": cid,
            "X": Xi,
            "y": yi,
            "num_samples": int(len(yi)),
            "class_hist": class_hist,
            "visible_classes": sorted([int(c) for c in client_class_sets[cid]]),
        })

    return clients


def train_local_model(
    X_local: np.ndarray,
    y_local: np.ndarray,
    feat_dim: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_epoch_log_lines: int = 10,
    verbose: bool = False,
    use_dpsgd: bool = False,
    dpsgd_batch_size: int = 64,
    dpsgd_clipping_norm: float = 1.0,
    dpsgd_noise_multiplier: float = 1.0,
) -> Dict:
    """本地监督训练（支持 plain SGD / 标准 DPSGD）。"""
    rng = np.random.RandomState(seed)

    X = np.asarray(X_local, dtype=np.float32)
    y = np.asarray(y_local).reshape(-1)
    n = len(y)
    perm = rng.permutation(n)

    n_val = int(max(1, round(n * val_ratio))) if n > 5 else 0
    if n_val >= n:
        n_val = max(0, n - 1)

    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    if n_val > 0:
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        X_val, y_val = None, None

    model = make_model(feat_dim, num_classes, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    tx = to_tensor_x(X_tr, device)
    ty = to_tensor_y(y_tr, num_classes, device)

    log_epochs = build_log_epochs(epochs, max_lines=max_epoch_log_lines)
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        if not use_dpsgd:
            optimizer.zero_grad()
            logits = model(tx)
            if num_classes == 2:
                loss = criterion(logits, ty)
            else:
                loss = criterion(logits, ty.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss = float(loss.detach().cpu().item())
        else:
            # 标准 DPSGD（样本级梯度裁剪 + 高斯噪声）
            # 这里按 mini-batch 执行，便于和论文里的 DP 训练流程对齐。
            batch_size = max(1, int(dpsgd_batch_size))
            clip_c = float(max(1e-8, dpsgd_clipping_norm))
            noise_mul = float(max(0.0, dpsgd_noise_multiplier))

            order = rng.permutation(len(y_tr))
            num_batches = 0

            for st in range(0, len(order), batch_size):
                ed = min(st + batch_size, len(order))
                bidx = order[st:ed]
                xb = to_tensor_x(X_tr[bidx], device)
                yb = to_tensor_y(y_tr[bidx], num_classes, device)
                bsz = int(len(bidx))
                if bsz <= 0:
                    continue

                params = [p for p in model.parameters() if p.requires_grad]
                summed = [torch.zeros_like(p) for p in params]

                batch_loss = 0.0
                for i in range(bsz):
                    optimizer.zero_grad()
                    xi = xb[i:i + 1]
                    yi = yb[i:i + 1]
                    logits_i = model(xi)
                    if num_classes == 2:
                        loss_i = criterion(logits_i, yi)
                    else:
                        loss_i = criterion(logits_i, yi.view(-1))
                    loss_i.backward()
                    batch_loss += float(loss_i.detach().cpu().item())

                    grads_i = [p.grad.detach().clone() for p in params]
                    flat = torch.cat([g.reshape(-1) for g in grads_i], dim=0)
                    g_norm = float(torch.norm(flat, p=2).detach().cpu().item())
                    coef = min(1.0, clip_c / (g_norm + 1e-12))
                    for gi, acc in zip(grads_i, summed):
                        acc.add_(gi * coef)

                optimizer.zero_grad()
                for p, acc in zip(params, summed):
                    # 对“和梯度”加噪，再除以 batch_size，等价于均值梯度上的噪声
                    if noise_mul > 0.0:
                        noise = torch.normal(
                            mean=0.0,
                            std=noise_mul * clip_c,
                            size=acc.shape,
                            device=acc.device,
                            dtype=acc.dtype,
                        )
                        acc = acc + noise
                    p.grad = acc / float(bsz)
                optimizer.step()

                epoch_loss += batch_loss / float(bsz)
                num_batches += 1

            if num_batches > 0:
                epoch_loss /= float(num_batches)

        train_acc = evaluate_model(model, X_tr, y_tr, num_classes, device)
        val_acc = evaluate_model(model, X_val, y_val, num_classes, device) if X_val is not None else train_acc

        if ep in log_epochs:
            row = {
                "epoch": ep,
                "loss": float(epoch_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "train_mode": "dpsgd" if use_dpsgd else "sgd",
            }
            history.append(row)
            if verbose:
                print("local ep={:>4d}/{:<4d} loss={:.6f} train={:.4f} val={:.4f}".format(
                    ep, epochs, row["loss"], row["train_acc"], row["val_acc"]
                ))

    final_train_acc = evaluate_model(model, X_tr, y_tr, num_classes, device)
    final_val_acc = evaluate_model(model, X_val, y_val, num_classes, device) if X_val is not None else final_train_acc

    return {
        "model": model,
        "train_acc": float(final_train_acc),
        "val_acc": float(final_val_acc),
        "history": history,
        "num_train": int(len(y_tr)),
        "num_val": int(n_val),
        "train_mode": "dpsgd" if use_dpsgd else "sgd",
    }


def build_reference_set(
    clients: List[Dict],
    ref_size: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """从各 client 均匀抽样构建 reference set。"""
    rng = np.random.RandomState(seed)

    num_clients = len(clients)
    quota = max(1, ref_size // max(1, num_clients))

    xs, ys = [], []
    taken = 0
    for c in clients:
        Xi, yi = c["X"], c["y"]
        n = len(yi)
        if n == 0:
            continue
        k = min(quota, n)
        idx = rng.choice(n, size=k, replace=False)
        xs.append(Xi[idx])
        ys.append(yi[idx])
        taken += k

    if taken < ref_size:
        X_pool = np.concatenate([c["X"] for c in clients], axis=0)
        y_pool = np.concatenate([c["y"] for c in clients], axis=0)
        remain = min(ref_size - taken, len(y_pool))
        idx = rng.choice(len(y_pool), size=remain, replace=False)
        xs.append(X_pool[idx])
        ys.append(y_pool[idx])

    X_ref = np.concatenate(xs, axis=0).astype(np.float32)
    y_ref = np.concatenate(ys, axis=0).reshape(-1)

    if len(y_ref) > ref_size:
        idx = rng.choice(len(y_ref), size=ref_size, replace=False)
        X_ref, y_ref = X_ref[idx], y_ref[idx]

    return X_ref, y_ref


def get_soft_predictions(
    model: nn.Module,
    X: np.ndarray,
    num_classes: int,
    device: torch.device,
    tau: float = 2.0,
    batch_size: int = 1024,
) -> np.ndarray:
    """获取模型在输入上的 soft predictions。"""
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    probs_all = []

    with torch.no_grad():
        for st in range(0, len(X), batch_size):
            ed = min(st + batch_size, len(X))
            xb = to_tensor_x(X[st:ed], device)
            logits = model(xb)
            probs = logits_to_probs2(logits, num_classes=num_classes, tau=tau)
            probs_all.append(probs.detach().cpu().numpy())

    return np.concatenate(probs_all, axis=0)


def build_ensemble_teacher(
    local_models: List[nn.Module],
    X_ref: np.ndarray,
    num_classes: int,
    device: torch.device,
    tau: float = 2.0,
) -> np.ndarray:
    """平均 teacher：对所有 local 模型 soft prediction 取简单平均。"""
    all_probs = []
    for m in local_models:
        p = get_soft_predictions(m, X_ref, num_classes, device, tau=tau)
        all_probs.append(p)
    teacher_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    return teacher_probs.astype(np.float32)


def distill_local_model(
    model: nn.Module,
    X_local: np.ndarray,
    y_local: np.ndarray,
    X_ref: np.ndarray,
    teacher_probs_ref: np.ndarray,
    num_classes: int,
    device: torch.device,
    tau: float = 2.0,
    lambda_kd: float = 1.0,
    distill_epochs: int = 30,
    lr: float = 5e-3,
    max_epoch_log_lines: int = 10,
    verbose: bool = False,
) -> Dict:
    """在本地监督 loss 基础上加入 KD loss。"""
    student = copy.deepcopy(model).to(device)

    criterion_sup = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(student.parameters(), lr=lr)

    tx_local = to_tensor_x(X_local, device)
    ty_local = to_tensor_y(y_local, num_classes, device)
    tx_ref = to_tensor_x(X_ref, device)
    t_ref = torch.from_numpy(np.asarray(teacher_probs_ref, dtype=np.float32)).to(device)

    before_acc = evaluate_model(student, X_local, y_local, num_classes, device)
    log_epochs = build_log_epochs(distill_epochs, max_lines=max_epoch_log_lines)
    history = []

    for ep in range(1, distill_epochs + 1):
        student.train()
        optimizer.zero_grad()

        logits_local = student(tx_local)
        if num_classes == 2:
            sup_loss = criterion_sup(logits_local, ty_local)
        else:
            sup_loss = criterion_sup(logits_local, ty_local.view(-1))

        logits_ref = student(tx_ref)
        s_probs = logits_to_probs2(logits_ref, num_classes=num_classes, tau=tau)
        s_log_probs = torch.log(torch.clamp(s_probs, min=1e-8))
        kd_loss = criterion_kd(s_log_probs, torch.clamp(t_ref, min=1e-8)) * (tau * tau)

        loss = sup_loss + float(lambda_kd) * kd_loss
        loss.backward()
        optimizer.step()

        if ep in log_epochs:
            after_ep_acc = evaluate_model(student, X_local, y_local, num_classes, device)
            row = {
                "epoch": ep,
                "sup_loss": float(sup_loss.detach().cpu().item()),
                "kd_loss": float(kd_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
                "local_acc": float(after_ep_acc),
            }
            history.append(row)
            if verbose:
                print("distill ep={:>4d}/{:<4d} total={:.6f} sup={:.6f} kd={:.6f} acc={:.4f}".format(
                    ep, distill_epochs, row["total_loss"], row["sup_loss"], row["kd_loss"], row["local_acc"]
                ))

    after_acc = evaluate_model(student, X_local, y_local, num_classes, device)

    return {
        "model": student,
        "before_acc": float(before_acc),
        "after_acc": float(after_acc),
        "history": history,
    }


def average_model_weights(models: List[nn.Module], feat_dim: int, num_classes: int,
                          device: torch.device) -> nn.Module:
    """参数平均初始化。"""
    assert len(models) > 0, "models 不能为空"
    avg_model = make_model(feat_dim, num_classes, device)

    state_dicts = [m.state_dict() for m in models]
    avg_state = {}
    for k in state_dicts[0].keys():
        avg_state[k] = sum(sd[k] for sd in state_dicts) / float(len(state_dicts))

    avg_model.load_state_dict(avg_state)
    return avg_model


def select_best_local_model(models: List[nn.Module], X_ref: np.ndarray, y_ref: np.ndarray,
                            num_classes: int, device: torch.device) -> Tuple[nn.Module, int, float]:
    """在 reference set 上选择最好 local 模型。"""
    best_idx, best_acc = -1, -1.0
    for i, m in enumerate(models):
        acc = evaluate_model(m, X_ref, y_ref, num_classes, device)
        if acc > best_acc:
            best_acc = acc
            best_idx = i
    return copy.deepcopy(models[best_idx]).to(device), int(best_idx), float(best_acc)


def evaluate_candidate_models(
    avg_model: nn.Module,
    best_model: nn.Module,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> Dict:
    avg_acc = evaluate_model(avg_model, X_ref, y_ref, num_classes, device)
    best_acc = evaluate_model(best_model, X_ref, y_ref, num_classes, device)

    if avg_acc >= best_acc:
        chosen = "avg"
        chosen_model = avg_model
    else:
        chosen = "best"
        chosen_model = best_model

    return {
        "avg_ref_acc": float(avg_acc),
        "best_ref_acc": float(best_acc),
        "chosen_init_type": chosen,
        "chosen_model": chosen_model,
    }


def compute_supervised_loss(model: nn.Module, x: np.ndarray, y: np.ndarray,
                            num_classes: int, device: torch.device) -> float:
    """计算监督损失（仅用于评估曲线，不参与反向传播）。"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    tx = to_tensor_x(x, device)
    ty = to_tensor_y(y, num_classes, device)
    with torch.no_grad():
        logits = model(tx)
        if num_classes == 2:
            loss = criterion(logits, ty)
        else:
            loss = criterion(logits, ty.view(-1))
    return float(loss.detach().cpu().item())


def train_plain_global_from_init(
    init_model: nn.Module,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-2,
    max_epoch_log_lines: int = 10,
    verbose: bool = False,
) -> Dict:
    """
    从给定初始化出发执行 plain global training（无 KD，纯监督）。
    返回可直接用于画 convergence curve 的历史数据。
    """
    model = copy.deepcopy(init_model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    tx = to_tensor_x(X_train_feat, device)
    ty = to_tensor_y(y_train, num_classes, device)
    log_epochs = build_log_epochs(epochs, max_lines=max_epoch_log_lines)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(tx)
        if num_classes == 2:
            loss = criterion(logits, ty)
        else:
            loss = criterion(logits, ty.view(-1))
        loss.backward()
        optimizer.step()

        train_acc = evaluate_model(model, X_train_feat, y_train, num_classes, device)
        test_acc = evaluate_model(model, X_test_feat, y_test, num_classes, device)
        train_loss = compute_supervised_loss(model, X_train_feat, y_train, num_classes, device)
        test_loss = compute_supervised_loss(model, X_test_feat, y_test, num_classes, device)

        row = {
            "epoch": ep,
            "iteration": ep,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
        }
        history.append(row)
        if verbose and ep in log_epochs:
            print("global it={:>4d}/{:<4d} tr_loss={:.6f} te_loss={:.6f} tr_acc={:.4f} te_acc={:.4f}".format(
                ep, epochs, train_loss, test_loss, train_acc, test_acc
            ))

    return {
        "model": model,
        "history": history,
        "final_train_acc": float(history[-1]["train_acc"]),
        "final_test_acc": float(history[-1]["test_acc"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_test_loss": float(history[-1]["test_loss"]),
    }


def run_figure5_init_comparison(
    feat_dim: int,
    num_classes: int,
    avg_model: nn.Module,
    best_model: nn.Module,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    global_epochs: int = 80,
    global_lr: float = 1e-2,
    seed: int = 42,
    max_epoch_log_lines: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Figure5 对应逻辑：
      random / avg / best 三种初始化分别做 plain global training，
      对比其 convergence curve。
    """
    set_seed(seed)
    random_model = make_model(feat_dim, num_classes, device)

    if verbose:
        print("\n[Figure5] 初始化对比：random vs avg vs best（plain global training）")
        print("  global_epochs/lr  :", global_epochs, global_lr)

    candidates = {
        "random_init": random_model,
        "avg_init": avg_model,
        "best_init": best_model,
    }

    curves = {}
    finals = {}
    for name, init_m in candidates.items():
        if verbose:
            print(f"\n  -> 训练 {name}")
        out = train_plain_global_from_init(
            init_model=init_m,
            X_train_feat=X_train_feat,
            y_train=y_train,
            X_test_feat=X_test_feat,
            y_test=y_test,
            num_classes=num_classes,
            device=device,
            epochs=global_epochs,
            lr=global_lr,
            max_epoch_log_lines=max_epoch_log_lines,
            verbose=verbose,
        )
        curves[name] = out["history"]
        finals[name] = {
            "final_train_acc": out["final_train_acc"],
            "final_test_acc": out["final_test_acc"],
            "final_train_loss": out["final_train_loss"],
            "final_test_loss": out["final_test_loss"],
        }

    winner = max(finals.keys(), key=lambda k: finals[k]["final_test_acc"])
    if verbose:
        print("\n[Figure5] 最终对比（按 test_acc）:")
        for k in ["random_init", "avg_init", "best_init"]:
            print("  {:<12} test_acc={:.4f} train_acc={:.4f}".format(
                k, finals[k]["final_test_acc"], finals[k]["final_train_acc"]
            ))
        print("  winner:", winner)

    return {
        "config": {
            "global_epochs": int(global_epochs),
            "global_lr": float(global_lr),
            "seed": int(seed),
        },
        "curves": curves,
        "finals": finals,
        "winner": winner,
    }


def run_kd_hetero_init_pipeline(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    num_clients: int = 5,
    classes_per_client: Optional[int] = None,
    ref_size: int = 200,
    local_epochs: int = 80,
    local_lr: float = 1e-2,
    tau: float = 2.0,
    lambda_kd: float = 1.0,
    distill_epochs: int = 30,
    distill_lr: float = 5e-3,
    seed: int = 42,
    max_epoch_log_lines: int = 10,
    run_figure5: bool = False,
    global_epochs: int = 80,
    global_lr: float = 1e-2,
    device: Optional[str] = None,
    dataset_name: str = "unknown",
    save_init_model: bool = True,
    use_secure_aggregation: bool = True,
    secure_max_threads: int = 2,
    local_use_dpsgd: bool = False,
    local_dpsgd_batch_size: int = 64,
    local_dpsgd_clipping_norm: float = 1.0,
    local_dpsgd_noise_multiplier: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """完整执行模块 b。"""
    set_seed(seed)

    X_train_feat = np.asarray(X_train_feat, dtype=np.float32)
    X_test_feat = np.asarray(X_test_feat, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    feat_dim = int(X_train_feat.shape[1])
    num_classes = infer_num_classes(y_train)

    if device is None:
        # 当前实验环境里 torch 与 RTX4090 的 CUDA 架构不兼容，默认走 CPU 更稳。
        device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    if verbose:
        print("=" * 72)
        print("模块 b：KD-based Heterogeneity-aware Initialization（最简版）")
        print("=" * 72)
        print("feat_dim            :", feat_dim)
        print("num_classes         :", num_classes)
        print("num_clients         :", num_clients)
        print("classes_per_client  :", classes_per_client)
        print("ref_size            :", ref_size)
        print("local_epochs/lr     :", local_epochs, local_lr)
        print("distill_epochs/lr   :", distill_epochs, distill_lr)
        print("tau/lambda_kd       :", tau, lambda_kd)
        print("local_train_mode    :", "DPSGD" if local_use_dpsgd else "SGD")
        if local_use_dpsgd:
            print("dpsgd(batch/C/sigma): {}/{}/{}".format(
                int(local_dpsgd_batch_size),
                float(local_dpsgd_clipping_norm),
                float(local_dpsgd_noise_multiplier),
            ))
        print("run_figure5         :", run_figure5)
        if run_figure5:
            print("global_epochs/lr    :", global_epochs, global_lr)
        print("device              :", device_obj)

    clients = make_class_skew_splits(
        X_train_feat, y_train,
        num_clients=num_clients,
        classes_per_client=classes_per_client,
        seed=seed,
    )

    if verbose:
        print("\n[Step2] class-skew 划分统计")
        for c in clients:
            print("  client {:>2d} | n={:<5d} | classes={} | hist={}".format(
                c["client_id"], c["num_samples"], c["visible_classes"], c["class_hist"]
            ))

    local_models = []
    local_logs = []
    if verbose:
        print("\n[Step3] local pretraining")

    for c in clients:
        out = train_local_model(
            c["X"], c["y"],
            feat_dim=feat_dim,
            num_classes=num_classes,
            device=device_obj,
            epochs=local_epochs,
            lr=local_lr,
            seed=seed + c["client_id"],
            max_epoch_log_lines=max_epoch_log_lines,
            use_dpsgd=bool(local_use_dpsgd),
            dpsgd_batch_size=int(local_dpsgd_batch_size),
            dpsgd_clipping_norm=float(local_dpsgd_clipping_norm),
            dpsgd_noise_multiplier=float(local_dpsgd_noise_multiplier),
            verbose=False,
        )
        local_models.append(out["model"])
        local_logs.append({
            "client_id": c["client_id"],
            "train_acc": out["train_acc"],
            "val_acc": out["val_acc"],
            "history": out["history"],
            "num_train": out["num_train"],
            "num_val": out["num_val"],
        })
        if verbose:
            print("  client {:>2d} | train_acc={:.4f} | val_acc={:.4f}".format(
                c["client_id"], out["train_acc"], out["val_acc"]
            ))

    X_ref, y_ref = build_reference_set(clients, ref_size=ref_size, seed=seed)
    if verbose:
        print("\n[Step4] reference set: X_ref={}, y_ref={}".format(X_ref.shape, y_ref.shape))

    teacher_probs = build_ensemble_teacher(
        local_models, X_ref,
        num_classes=num_classes,
        device=device_obj,
        tau=tau,
    )
    if verbose:
        print("[Step5] ensemble teacher probs shape:", teacher_probs.shape)

    distill_models = []
    distill_logs = []
    if verbose:
        print("\n[Step6] KD distillation")

    for c, base_model in zip(clients, local_models):
        d_out = distill_local_model(
            base_model,
            X_local=c["X"],
            y_local=c["y"],
            X_ref=X_ref,
            teacher_probs_ref=teacher_probs,
            num_classes=num_classes,
            device=device_obj,
            tau=tau,
            lambda_kd=lambda_kd,
            distill_epochs=distill_epochs,
            lr=distill_lr,
            max_epoch_log_lines=max_epoch_log_lines,
            verbose=False,
        )
        distill_models.append(d_out["model"])
        distill_logs.append({
            "client_id": c["client_id"],
            "before_acc": d_out["before_acc"],
            "after_acc": d_out["after_acc"],
            "history": d_out["history"],
        })
        if verbose:
            print("  client {:>2d} | before={:.4f} -> after={:.4f}".format(
                c["client_id"], d_out["before_acc"], d_out["after_acc"]
            ))

    # ------------------------------------------------------------------
    # 聚合 + 候选评估：密态（ABY3）或明文
    # ------------------------------------------------------------------
    # best_local_model: ref set 上精度最高的单个蒸馏后 client 模型（明文选出，作为候选之一）
    best_model_plain, best_idx, _ = select_best_local_model(
        distill_models,
        X_ref=X_ref,
        y_ref=y_ref,
        num_classes=num_classes,
        device=device_obj,
    )

    _use_secure = bool(use_secure_aggregation)
    if _use_secure:
        try:
            import secure_aggregation as sa
        except ImportError:
            if verbose:
                print("[警告] secure_aggregation.py 不可用，回退到明文聚合")
            _use_secure = False

    if _use_secure:
        if verbose:
            print("\n[Step7] 使用 ABY3 密态聚合 + 密态候选评估 ...")
        sa_result = sa.secure_aggregate_and_select(
            distill_models=distill_models,
            best_local_model=best_model_plain,
            X_ref=X_ref,
            y_ref=y_ref,
            feat_dim=feat_dim,
            num_classes=num_classes,
            max_threads=secure_max_threads,
            verbose=verbose,
        )
        avg_model = sa_result["avg_model"].to(device_obj)
        best_model = sa_result["best_model"].to(device_obj)
        chosen_model = sa_result["chosen_model"].to(device_obj)
        cand = {
            "avg_ref_acc": sa_result["candidate"]["avg_ref_acc"],
            "best_ref_acc": sa_result["candidate"]["best_ref_acc"],
            "chosen_init_type": sa_result["candidate"]["chosen_init_type"],
            "chosen_model": chosen_model,
            "secure_aggregation": True,
        }
    else:
        if verbose:
            print("\n[Step7] 使用明文聚合 + 明文候选评估 ...")
        avg_model = average_model_weights(
            distill_models,
            feat_dim=feat_dim,
            num_classes=num_classes,
            device=device_obj,
        )
        best_model = best_model_plain
        cand = evaluate_candidate_models(
            avg_model=avg_model,
            best_model=best_model,
            X_ref=X_ref,
            y_ref=y_ref,
            num_classes=num_classes,
            device=device_obj,
        )
        chosen_model = cand["chosen_model"]
        cand["secure_aggregation"] = False

    avg_test_acc = evaluate_model(avg_model, X_test_feat, y_test, num_classes, device_obj)
    best_test_acc = evaluate_model(best_model, X_test_feat, y_test, num_classes, device_obj)
    chosen_test_acc = evaluate_model(chosen_model, X_test_feat, y_test, num_classes, device_obj)

    # 说明：
    # 旧版 feature_kd_init.py 里的 Figure5 是明文近似实现。
    # 为与 PEA 原文对齐，Figure5 已迁移到模块 C 中的 TFE 密态训练流程执行。
    figure5 = None
    if run_figure5 and verbose:
        print("\n[Figure5] 已从模块 b 明文近似实现迁移到模块 c 的 TFE 密态训练实现。")
        print("          当前模块 b 不再生成 Figure5 曲线。")

    if verbose:
        agg_mode = "ABY3 密态" if cand.get("secure_aggregation", False) else "明文"
        print(f"\n[Step8] candidate comparison ({agg_mode})")
        print("  avg_ref_acc   : {:.4f}".format(cand["avg_ref_acc"]))
        print("  best_ref_acc  : {:.4f}".format(cand["best_ref_acc"]))
        print("  best_local_id : {}".format(best_idx))
        print("  chosen        : {}".format(cand["chosen_init_type"]))
        print("  aggregation   : {}".format(agg_mode))
        print("\n  [test set check]")
        print("  avg_test_acc  : {:.4f}".format(avg_test_acc))
        print("  best_test_acc : {:.4f}".format(best_test_acc))
        print("  chosen_test_acc: {:.4f}".format(chosen_test_acc))

    config = {
        "num_clients": num_clients,
        "classes_per_client": classes_per_client,
        "ref_size": ref_size,
        "local_epochs": local_epochs,
        "local_lr": local_lr,
        "tau": tau,
        "lambda_kd": lambda_kd,
        "distill_epochs": distill_epochs,
        "distill_lr": distill_lr,
        "seed": seed,
        "device": str(device_obj),
        "run_figure5": bool(run_figure5),
        "global_epochs": int(global_epochs),
        "global_lr": float(global_lr),
        "use_secure_aggregation": bool(cand.get("secure_aggregation", False)),
        "local_use_dpsgd": bool(local_use_dpsgd),
        "local_dpsgd_batch_size": int(local_dpsgd_batch_size),
        "local_dpsgd_clipping_norm": float(local_dpsgd_clipping_norm),
        "local_dpsgd_noise_multiplier": float(local_dpsgd_noise_multiplier),
    }

    init_model_cache = None
    if save_init_model:
        init_model_cache = _save_module_b_init_cache(
            dataset_name=dataset_name,
            feat_dim=feat_dim,
            num_classes=num_classes,
            candidate={
                "avg_ref_acc": cand["avg_ref_acc"],
                "best_ref_acc": cand["best_ref_acc"],
                "best_local_idx": best_idx,
                "chosen_init_type": cand["chosen_init_type"],
                "secure_aggregation": bool(cand.get("secure_aggregation", False)),
                "avg_test_acc": float(avg_test_acc),
                "best_test_acc": float(best_test_acc),
                "chosen_test_acc": float(chosen_test_acc),
            },
            config=config,
            chosen_model=chosen_model,
            avg_model=avg_model,
            best_model=best_model,
            x_train_shape=tuple(X_train_feat.shape),
            x_test_shape=tuple(X_test_feat.shape),
            y_train_shape=tuple(y_train.shape),
            y_test_shape=tuple(y_test.shape),
        )
        if verbose:
            print("\n[缓存] 模块 b 初始化模型已保存")
            print("  latest_pt :", init_model_cache["latest_pt"])

    return {
        "config": {
            **config,
            "dataset_name": str(dataset_name),
            "save_init_model": bool(save_init_model),
        },
        "split_stats": [
            {
                "client_id": c["client_id"],
                "num_samples": c["num_samples"],
                "visible_classes": c["visible_classes"],
                "class_hist": c["class_hist"],
            }
            for c in clients
        ],
        "local_logs": local_logs,
        "distill_logs": distill_logs,
        "reference": {
            "shape": tuple(X_ref.shape),
            "size": int(len(y_ref)),
        },
        # 供后续 Figure5 / 消融实验复用：不改变主流程，只额外暴露当次 reference set 和蒸馏后 local models。
        "reference_data": {
            "X_ref": X_ref,
            "y_ref": y_ref,
        },
        "candidate": {
            "avg_ref_acc": cand["avg_ref_acc"],
            "best_ref_acc": cand["best_ref_acc"],
            "best_local_idx": best_idx,
            "chosen_init_type": cand["chosen_init_type"],
            "secure_aggregation": bool(cand.get("secure_aggregation", False)),
            "avg_test_acc": float(avg_test_acc),
            "best_test_acc": float(best_test_acc),
            "chosen_test_acc": float(chosen_test_acc),
        },
        "initial_global_model": chosen_model,
        "avg_model": avg_model,
        "best_model": best_model,
        "distill_models": distill_models,
        "figure5": figure5,
        "init_model_cache": init_model_cache,
    }


def _demo_synthetic_run():
    """用合成特征做最小可运行验证。"""
    set_seed(42)

    n_train, n_test = 1200, 300
    feat_dim, num_classes = 64, 4

    W = np.random.randn(feat_dim, num_classes).astype(np.float32)
    X_train = np.random.randn(n_train, feat_dim).astype(np.float32)
    X_test = np.random.randn(n_test, feat_dim).astype(np.float32)
    y_train = np.argmax(X_train.dot(W) + 0.5 * np.random.randn(n_train, num_classes), axis=1)
    y_test = np.argmax(X_test.dot(W) + 0.5 * np.random.randn(n_test, num_classes), axis=1)

    out = run_kd_hetero_init_pipeline(
        X_train_feat=X_train,
        y_train=y_train,
        X_test_feat=X_test,
        y_test=y_test,
        num_clients=5,
        classes_per_client=2,
        ref_size=200,
        local_epochs=60,
        local_lr=0.02,
        tau=2.0,
        lambda_kd=0.8,
        distill_epochs=20,
        distill_lr=0.01,
        seed=42,
        max_epoch_log_lines=10,
        verbose=True,
    )

    print("\nDemo finished. chosen init =", out["candidate"]["chosen_init_type"])


if __name__ == "__main__":
    _demo_synthetic_run()
