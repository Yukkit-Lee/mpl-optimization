"""Figure 6 CIFAR-100 Fixed/Adaptive 用户可编辑参数入口。

这里只控制 CIFAR-100，不会改写 CIFAR-10 或 Tiny 的结果。Fixed 和 Adaptive
可以使用不同的 steps、学习率、batch 与裁剪参数；公平比较只强制二者使用相同
``target_epsilon``、``delta``、输入、开发集划分与 RNG seeds。

零训练检查：

    /root/autodl-tmp/new_envs/distillation/bin/python \
      new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
      --dataset-config cifar100 --check-only

正式三 seed（建议放入 tmux，完整命令见参数指南）：

    /root/autodl-tmp/new_envs/distillation/bin/python \
      new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
      --dataset-config cifar100 --workers 6
"""

# --------------------------- 输入数据（通常不要改） ---------------------------

# CIFAR-100 的 Stage-B C5 初始化器。若改成其他模型，实验问题和可比边界会改变。
INITIALIZER = (
    "new_output/new_stage_b_anchor_postagg/tfe_validation/cifar100/seed42/"
    "new_stage_b_anchor_postagg_tfe_model_cifar100_seed42.npz"
)

# 只包含训练与 development-validation 特征的 CIFAR-100 Stage-C 缓存。
FEATURE_CACHE = (
    "new_cache/new_true_ours/official_mtt/stage_c/cifar100/"
    "new_true_ours_stage_c_cifar100_seed42.npz"
)

DATASET = "cifar100"          # 写入结果 provenance；不要改成不存在的数据集名称。
RNG_SEEDS = [42, 43, 44]      # 正式结果使用三 seed。
# RNG_SEEDS = [17]             # 调参时可临时单 seed；不能据此画误差条。


# --------------------------- Fixed DP 可调参数 -------------------------------

FIXED_CONFIG = {
    "mode": "fixed_dp",          # 必须保持 fixed_dp。
    "steps": 280,                # 私有更新次数；增大后会重新校准噪声。
    "learning_rate": 0.013,      # Fixed 自己的学习率，可与 Adaptive 不同。
    "expected_batch_size": 16,   # Poisson 期望 batch，会影响采样率和隐私会计。
    "clip_c": 2.0,               # Fixed 全程使用的裁剪阈值 C。
    "target_epsilon": 3.0,       # 必须与 Adaptive 相同才是同隐私预算比较。
    "delta": 1e-5,               # 必须与 Adaptive 相同。

    # 以下字段仅用于统一配置结构，Fixed 不会用它们更新 C。
    "gamma": 0.5,
    "eta_c": 0.02,
    "warmup_steps": 0,
    "update_interval": 5,
    "count_noise_multiplier": 8.0,
    "c_min": 0.05,
    "c_max": 4.0,
}


# -------------------------- Adaptive DP 可调参数 -----------------------------

ADAPTIVE_CONFIG = {
    "mode": "adaptive_dp",       # 必须保持 adaptive_dp。
    "steps": 160,                # Adaptive 的更新次数，可与 Fixed 不同。
    "learning_rate": 0.002,      # Adaptive 自己的学习率。
    "expected_batch_size": 32,   # Adaptive 自己的 Poisson 期望 batch。
    "clip_c": 2.0,               # 初始 C0，不是最终阈值。
    "gamma": 0.5,                # 目标 noisy fraction，范围[0,1]。
    "eta_c": 0.1,                # C_t 调整速度；过大可能震荡。
    "warmup_steps": 0,           # 前多少步保持 C0。
    "update_interval": 5,        # 每隔多少步私有更新一次 C_t。
    "count_noise_multiplier": 16.0,  # count 机制噪声，参与总 epsilon 会计。
    "c_min": 0.05,               # C_t 下界。
    "c_max": 4.0,                # C_t 上界。
    "target_epsilon": 3.0,       # 必须与 Fixed 相同。
    "delta": 1e-5,               # 必须与 Fixed 相同。
}


# --------------------------- 独立输出（建议每轮改名） -------------------------

# CIFAR-100 使用独立文件名，不会覆盖现有三数据集正式 JSON。
OUTPUT_JSON = (
    "new_log/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_cifar100_formal.json"
)
OUTPUT_CSV = (
    "new_expTab/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_cifar100_formal.csv"
)
