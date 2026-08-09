#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Fixed-DP / Adaptive-DP 的 Module C 消融实验队列。

本文件是消融实验的“参数编排入口”，真正的单条 TFE/ABY3 训练由
``new_stage_c_secure_dpsgd.py`` 执行。用户应在以下配置中修改参数：

    new_configs/new_public_mtt_mainline/new_mainline_config.json
    -> module_c_formal -> mode_parameters

``fixed_dp.datasets`` 与 ``adaptive_dp.datasets`` 可以分别设置每个数据集的：

    steps          更新步数
    batch_size     batch大小/隐私会计采样率
    learning_rate  学习率
    clip_c         Fixed的固定C，或Adaptive的初始C0

Adaptive 还可以单独设置 count_noise_multiplier、warmup_steps、
update_interval、gamma、eta_c、c_min 和 c_max。JSON本身不能写注释，所以
各字段含义集中写在本文件和 ``new_stage_c_secure_dpsgd.py`` 参数区。

只生成一个隔离队列（不开始实验）：

    /root/autodl-tmp/new_envs/distillation/bin/python \
        new_prepare_module_c_formal.py --run-tag user_ablation_v1

检查生成的 TSV 后，使用 tmux 启动，避免断联停止：

    tmux new -s module_c_user_ablation
    bash new_scripts/new_run_task_queue.sh tfe \
        new_configs/new_public_mtt_mainline/new_tfe_module_c_user_ablation_v1.tsv

按 Ctrl-b、d 离开 tmux；用 ``tmux attach -t module_c_user_ablation`` 返回。
每次更换参数必须使用新的 ``--run-tag``，这样不会覆盖或跳过旧正式结果。

公平性提醒：若论文声称“同预算比较”，两种方法应保持相同的 epsilon、delta、
seed、初始化和数据，通常也应保持相同 steps/batch。若分别设置 steps/batch，
可以称为“各自调优的最佳效果”，但必须同时报告不同的计算与采样预算。
"""

import argparse
import hashlib
import json
import os
import re

from new_public_mtt_mainline_common import ROOT, load_config


TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT / "new_configs/new_public_mtt_mainline/new_tfe_module_c_formal.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_module_c_formal_plan.json"
)
MODULE_B_REPORT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_module_b_convergence_formal_report.json"
)


def resolve_mode_parameters(formal, dataset, mode):
    """合并旧共享参数与新的按方法/数据集独立参数。

    新配置 ``mode_parameters`` 优先；保留旧字段作为向后兼容默认值。这样历史
    formal 队列在数值上不变，但用户可以只修改 Fixed 或 Adaptive 的某一项。
    """
    legacy_dataset = formal["datasets"][dataset]
    mode_config = formal.get("mode_parameters", {}).get(mode, {})
    dataset_config = mode_config.get("datasets", {}).get(dataset, {})
    values = {
        "steps": int(dataset_config.get("steps", legacy_dataset["steps"])),
        "batch_size": int(
            dataset_config.get("batch_size", legacy_dataset["batch_size"])
        ),
        "learning_rate": float(
            dataset_config.get("learning_rate", formal["learning_rate"])
        ),
        "clip_c": float(dataset_config.get("clip_c", formal["clip_c"])),
        # 以下参数只会在 adaptive_dp 中实际更新C；Fixed传入后不会触发更新。
        "count_noise_multiplier": float(
            mode_config.get(
                "count_noise_multiplier", formal["count_noise_multiplier"]
            )
        ),
        "warmup_steps": int(
            mode_config.get("warmup_steps", formal["warmup_steps"])
        ),
        "update_interval": int(
            mode_config.get("update_interval", formal["update_interval"])
        ),
        "gamma": float(mode_config.get("gamma", 0.5)),
        "eta_c": float(mode_config.get("eta_c", 0.05)),
        "c_min": float(mode_config.get("c_min", 0.05)),
        "c_max": float(mode_config.get("c_max", 4.0)),
    }
    if values["steps"] <= 0 or values["batch_size"] <= 0:
        raise ValueError("steps和batch_size必须大于0: {} {}".format(dataset, mode))
    if values["learning_rate"] <= 0.0 or values["clip_c"] <= 0.0:
        raise ValueError("learning_rate和clip_c必须大于0")
    if not 0.0 < values["gamma"] < 1.0:
        raise ValueError("Adaptive gamma必须位于(0,1)")
    if not 0.0 < values["c_min"] <= values["clip_c"] <= values["c_max"]:
        raise ValueError("必须满足0<c_min<=clip_c<=c_max")
    return values


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epsilon_tag(value):
    return "eps{}".format(str(float(value)).replace(".", "p"))


def main():
    parser = argparse.ArgumentParser(
        description="生成可分别配置Fixed/Adaptive参数的Module C消融队列"
    )
    parser.add_argument("--activate", action="store_true")
    parser.add_argument(
        "--run-tag",
        default="formal",
        help=(
            "输出隔离标签；默认formal保持旧路径。自行调参时必须使用新标签，"
            "例如user_ablation_v1。"
        ),
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        raise ValueError("run-tag只能包含英文、数字、点、下划线和短横线")
    # 只有真正生成队列时才需要Opacus；延后导入可让普通Python环境正常查看
    # `--help` 和导入参数解析函数。正式运行请使用安装了Opacus的现代环境。
    from new_stage_c_privacy_calibration import adaptive_count_steps
    from new_stage_c_privacy_calibration import calibrate_noise_multiplier

    config = load_config()
    formal = config["module_c_formal"]
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    if formal["initializer_privacy_status"] != "data_independent":
        raise ValueError("Module C released-model DP requires data-independent init")
    if formal["test_loaded"]:
        raise ValueError("Module C formal preparation cannot load test")
    module_b = json.loads(MODULE_B_REPORT.read_text(encoding="utf-8"))
    if (
        module_b["status"] != "COMPLETE"
        or module_b["runs_completed"] != module_b["runs_expected"]
        or module_b["test_loaded"]
    ):
        raise ValueError("Module B convergence gate is incomplete")

    # formal保持历史路径；其他tag全部进入独立命名空间，防止覆盖旧54条结果。
    run_tag = args.run_tag
    queue = (
        QUEUE
        if run_tag == "formal"
        else ROOT
        / "new_configs/new_public_mtt_mainline/new_tfe_module_c_{}.tsv".format(
            run_tag
        )
    )
    plan = (
        PLAN
        if run_tag == "formal"
        else ROOT
        / "new_log/new_public_mtt_mainline/json/new_module_c_{}_plan.json".format(
            run_tag
        )
    )
    output_namespace = "module_c_formal" if run_tag == "formal" else "module_c_{}".format(run_tag)

    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    calibrations = {}
    for dataset in config["datasets"]:
        num_samples = {"cifar10": 400, "cifar100": 4000, "tiny": 8000}[dataset]
        for mode in formal["modes"]:
            # 关键变化：每种方法、每个数据集分别解析参数，不再强制共用。
            params = resolve_mode_parameters(formal, dataset, mode)
            steps = params["steps"]
            batch_size = params["batch_size"]
            sample_rate = float(batch_size) / num_samples
            count_steps = (
                adaptive_count_steps(
                    steps, params["warmup_steps"], params["update_interval"]
                )
                if mode == "adaptive_dp"
                else 0
            )
            for target_epsilon in formal["target_epsilons"]:
                noise, achieved = calibrate_noise_multiplier(
                    target_epsilon=target_epsilon,
                    steps=steps,
                    sample_rate=sample_rate,
                    delta=formal["delta"],
                    count_noise=(
                        params["count_noise_multiplier"] if count_steps else None
                    ),
                    count_steps=count_steps,
                )
                key = "{}:{}:{}".format(dataset, mode, target_epsilon)
                calibrations[key] = {
                    "dataset": dataset,
                    "mode": mode,
                    "target_epsilon": target_epsilon,
                    "achieved_epsilon": achieved,
                    "gradient_noise_multiplier": noise,
                    "count_noise_multiplier": (
                        params["count_noise_multiplier"] if count_steps else None
                    ),
                    "count_steps": count_steps,
                    "steps": steps,
                    "batch_size": batch_size,
                    "sample_rate": sample_rate,
                    "learning_rate": params["learning_rate"],
                    "clip_c": params["clip_c"],
                    "gamma": params["gamma"],
                    "eta_c": params["eta_c"],
                    "warmup_steps": params["warmup_steps"],
                    "update_interval": params["update_interval"],
                }
                for seed in formal["seeds"]:
                    cache = (
                        ROOT
                        / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
                        / dataset
                        / "new_stage_b_features_{}_seed{}.npz".format(
                            dataset, seed
                        )
                    )
                    initializer = (
                        ROOT
                        / "new_init_model/new_public_mtt_mainline/module_b_random"
                        / dataset
                        / "seed{}".format(seed)
                        / "new_random_initializer_{}_seed{}.npz".format(
                            dataset, seed
                        )
                    )
                    if not cache.is_file() or not initializer.is_file():
                        raise FileNotFoundError(
                            "missing formal asset for {} {}".format(dataset, seed)
                        )
                    tag = epsilon_tag(target_epsilon)
                    task_id = "public_mainline_{}_{}_{}_{}_seed{}".format(
                        output_namespace, dataset, mode, tag, seed
                    )
                    output_dir = (
                        ROOT
                        / "new_output/new_public_mtt_mainline"
                        / output_namespace
                        / dataset
                        / mode
                        / tag
                        / "seed{}".format(seed)
                    )
                    checkpoint = (
                        ROOT
                        / "new_checkpoint/new_public_mtt_mainline"
                        / output_namespace
                        / dataset
                        / mode
                        / tag
                        / "seed{}/new_stage_c".format(seed)
                    )
                    command = (
                        "{python} {script} --feature_cache {cache} "
                        "--initializer {initializer} --dataset {dataset} "
                        "--mode {mode} --output_dir {output} "
                        "--checkpoint_prefix {checkpoint} --seed {seed} "
                        "--batch_size {batch} --steps {steps} "
                        "--learning_rate {lr} --noise_multiplier {noise} "
                        "--count_noise_multiplier {count_noise} "
                        "--clip_c {clip_c} --warmup_steps {warmup} "
                        "--update_interval {interval} --gamma {gamma} "
                        "--eta_c {eta_c} --c_min {c_min} --c_max {c_max} "
                        "--checkpoint_interval {checkpoint_interval} "
                        "--initializer_privacy_status data_independent"
                    ).format(
                        python=TFE_PYTHON,
                        script=ROOT / "new_stage_c_secure_dpsgd.py",
                        cache=cache,
                        initializer=initializer,
                        dataset=dataset,
                        mode=mode,
                        output=output_dir,
                        checkpoint=checkpoint,
                        seed=seed,
                        batch=batch_size,
                        steps=steps,
                        lr=params["learning_rate"],
                        noise=noise,
                        count_noise=params["count_noise_multiplier"],
                        clip_c=params["clip_c"],
                        warmup=params["warmup_steps"],
                        interval=params["update_interval"],
                        gamma=params["gamma"],
                        eta_c=params["eta_c"],
                        c_min=params["c_min"],
                        c_max=params["c_max"],
                        checkpoint_interval=formal["checkpoint_interval"],
                    )
                    rows.append("{}\t{}\t{}".format(task_id, checkpoint, command))
                    tasks.append(
                        {
                            "task_id": task_id,
                            "phase": "module_c_formal",
                            "dataset": dataset,
                            "mode": mode,
                            "target_epsilon": target_epsilon,
                            "achieved_epsilon": achieved,
                            "gradient_noise_multiplier": noise,
                            "count_steps": count_steps,
                            "seed": seed,
                            "steps": steps,
                            "batch_size": batch_size,
                            "learning_rate": params["learning_rate"],
                            "clip_c": params["clip_c"],
                            "gamma": params["gamma"] if mode == "adaptive_dp" else "NOT_USED",
                            "eta_c": params["eta_c"] if mode == "adaptive_dp" else "NOT_USED",
                            "initializer": str(initializer),
                            "initializer_sha256": sha256_file(initializer),
                            "initializer_privacy_status": "data_independent",
                            "released_model_record_dp_valid": True,
                            "test_loaded": False,
                            "resource": "tfe",
                            "expected_output_dir": str(output_dir),
                        }
                    )
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("\n".join(rows) + "\n", encoding="utf-8")
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(
        json.dumps(
            {
                "schema": "new_module_c_formal_plan_v1",
                "run_tag": run_tag,
                "output_namespace": output_namespace,
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "module_b_gate": str(MODULE_B_REPORT),
                "module_b_gate_status": "COMPLETE",
                "privacy_scope": "stage_c_distilled_feature_records",
                "released_model_record_dp_valid": True,
                "initializer_privacy_status": "data_independent",
                "accountant": "Opacus RDPAccountant",
                "resource_policy": {
                    "resource": "tfe",
                    "cpu_affinity": "unrestricted",
                    "max_concurrent_tasks": 1,
                },
                "calibrations": calibrations,
                "queue": str(queue),
                "tasks": tasks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text("tfe\t{}\n".format(queue), encoding="utf-8")
    print(
        "phase=module_c_formal tasks={} calibrations={} test_loaded=False "
        "released_model_record_dp_valid=True".format(
            len(tasks), len(calibrations)
        )
    )
    print("queue={}".format(queue))
    print("plan={}".format(plan))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
