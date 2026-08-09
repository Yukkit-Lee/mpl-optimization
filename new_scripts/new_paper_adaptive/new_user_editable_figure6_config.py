"""Figure 6 消融实验的用户可编辑参数文件。

你通常只需要修改本文件，不需要改训练器。Fixed DP 和 Adaptive DP 的参数
完全分开，除了 ``target_epsilon`` 应保持相同以满足“相同隐私预算”口径外，
steps、学习率、batch 和裁剪参数都可以不同。

快速试跑（建议先把 RNG_SEEDS 临时改成 [17]）：

    tmux new -s figure6_tune
    cd /root/autodl-tmp/tfe_experiments
    python new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
        --workers 2

正式三 seed：把 RNG_SEEDS 改回 [42, 43, 44] 后运行同一命令。结果会写入
``new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.json``。
"""

# --------------------------- 输入数据（通常不要改） ---------------------------

# Stage-B C5 初始化模型。更换它意味着实验问题改变，不能再与当前 Figure 6 直接比较。
INITIALIZER = (
    "new_output/new_stage_b_anchor_postagg/tfe_validation/cifar10/seed42/"
    "new_stage_b_anchor_postagg_tfe_model_cifar10_seed42.npz"
)

# CIFAR-10 Stage-C 特征缓存；代码会拒绝读取包含 official test 的缓存。
FEATURE_CACHE = (
    "new_cache/new_true_ours/official_mtt/stage_c/cifar10/"
    "new_true_ours_stage_c_cifar10_seed42.npz"
)

# 正式论文图建议至少使用三个预先确定的随机种子。
# 调参阶段可以临时改成 [17] 以节省时间，但单 seed 只能用于诊断，不能报告误差条。
RNG_SEEDS = [42, 43, 44]


# --------------------------- Fixed DP 可调参数 -------------------------------

FIXED_CONFIG = {
    "mode": "fixed_dp",          # 方法类型；Fixed 必须保持 fixed_dp。
    "steps": 320,                # 私有梯度更新次数；越大通常更慢且会影响所需噪声。
    "learning_rate": 0.001,      # 学习率；过小收敛慢，过大可能震荡或精度下降。
    "expected_batch_size": 16,   # Poisson 采样的期望 batch；会改变采样率和隐私会计。
    "clip_c": 2.0,               # 固定裁剪阈值 C；Fixed 全程不更新该数值。
    "target_epsilon": 3.0,       # 目标隐私预算；公平比较时应与 Adaptive 完全相同。
    "delta": 1e-5,               # DP 的 delta；公平比较时也应与 Adaptive 相同。

    # 以下字段只为与统一训练器的数据结构兼容，Fixed 模式不会使用它们更新 C。
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
    "mode": "adaptive_dp",       # 方法类型；Adaptive 必须保持 adaptive_dp。
    "steps": 160,                # 可与 Fixed 不同；每次改变后都会重新校准梯度噪声。
    "learning_rate": 0.003,      # Adaptive 自己的学习率，不需要与 Fixed 相同。
    "expected_batch_size": 32,   # Adaptive 自己的期望 batch，也可与 Fixed 不同。
    "clip_c": 2.0,               # 初始裁剪阈值 C0，不是最终 C。
    "gamma": 0.5,                # 目标“低于阈值”的 noisy fraction，取值应在 0 到 1。
    "eta_c": 0.1,                # C 的更新幅度；越大适应越快，也越容易震荡。
    "warmup_steps": 0,           # 前多少步不更新 C。
    "update_interval": 5,        # 每隔多少步私有更新一次 C；越频繁 count release 越多。
    "count_noise_multiplier": 16.0,  # 私有 count 的噪声倍数；会参与总 epsilon 会计。
    "c_min": 0.05,               # 自适应阈值允许达到的最小值。
    "c_max": 4.0,                # 自适应阈值允许达到的最大值。
    "target_epsilon": 3.0,       # 必须与 Fixed 相同，除非你明确改变比较问题。
    "delta": 1e-5,               # 建议与 Fixed 相同。
}


# --------------------------- 输出位置（可以改名） ----------------------------

# 每次尝试不同参数时建议改文件名，避免覆盖此前结果。
OUTPUT_JSON = "new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.json"
OUTPUT_CSV = "new_expTab/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.csv"
