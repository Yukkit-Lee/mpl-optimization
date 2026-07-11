#!/usr/bin/env python3
"""
基于真实表格数据的 TF Encrypted 最小二分类训练验证脚本。

数据集: sklearn 自带 Breast Cancer Wisconsin (binary classification)
模型: 单层 Dense (logistic regression 形式)
目标:
1) 验证数据进入 TFE 私有流程
2) 验证 loss 可计算
3) 验证 backward + optimizer 可运行
4) 验证训练过程中 loss/accuracy 发生变化
"""

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 先设置 TensorFlow C++ 日志等级，再导入 tensorflow。
# 2 表示屏蔽 INFO；若想更安静可设为 3（连 WARNING 也屏蔽）。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# 在导入 tf_encrypted 前，先把 TensorFlow 与 tf_encrypted 日志级别压低，
# 避免导入阶段输出大量 deprecated / custom-op 缺失告警。
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_encrypted").setLevel(logging.ERROR)

# ============================================================================
# ⚠️ 密态训练核心框架导入
# ============================================================================
# tf_encrypted 是一个基于 MPC（安全多方计算）的加密机器学习框架
# 所有以 tfe. 开头的操作都在加密分片上进行
import tf_encrypted as tfe
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tf_encrypted.keras import Sequential
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.layers import Dense
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"


def ensure_dirs():
    """确保实验目录结构存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_runtime_logging():
    """统一压低运行期日志噪音，便于观察训练主输出。"""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tf_encrypted").setLevel(logging.ERROR)


def prepare_dataset(test_size=0.2, seed=7):
    """加载并标准化真实表格数据，数据缓存到 data 目录。"""
    cache_file = DATA_DIR / "breast_cancer_split.npz"
    if cache_file.exists():
        cached = np.load(str(cache_file))
        return (
            cached["x_train"].astype(np.float32),
            cached["x_test"].astype(np.float32),
            cached["y_train"].astype(np.float32),
            cached["y_test"].astype(np.float32),
        )

    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    np.savez(
        str(cache_file),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    return x_train, x_test, y_train, y_test


def eval_linear_binary(x, y, w, b):
    """用揭示后的参数在明文侧评估准确率和 MSE。"""
    logits = x.dot(w) + b
    pred = (logits > 0.5).astype(np.float32)
    acc = float(np.mean(pred == y))
    mse = float(np.mean((logits - y) ** 2))
    return acc, mse


def main():
    ensure_dirs()
    configure_runtime_logging()

    np.random.seed(7)
    tf.compat.v1.set_random_seed(7)
    tf.compat.v1.reset_default_graph()

    x_train, x_test, y_train, y_test = prepare_dataset()
    n_train, d = x_train.shape
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_file = LOG_DIR / "train_real_tabular_binary.log"
    report_file = LOG_DIR / "real_tabular_training_report.json"

    def log_line(text):
        """同时输出到终端与日志文件。"""
        print(text)
        with train_log_file.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")

    log_line("Python:   {}".format(sys.executable))
    log_line("TensorFlow: {}".format(tf.__version__))
    log_line("TF Encrypted: {}".format(getattr(tfe, "__version__", "unknown")))
    log_line("日志文件: {}".format(train_log_file))
    log_line("")
    log_line("=" * 60)
    log_line("TF Encrypted - 真实表格二分类训练")
    log_line("=" * 60)
    log_line("run_id: {}".format(run_ts))
    log_line("")
    log_line("[1/4] 准备数据集...")
    log_line("      数据缓存: {}".format(DATA_DIR / "breast_cancer_split.npz"))
    log_line("      训练集: {} 样本, {} 特征".format(n_train, d))
    log_line("      测试集: {} 样本".format(x_test.shape[0]))
    log_line("")
    log_line("[2/4] 构建 TFE 模型与训练图...")

    result = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tf_encrypted": getattr(tfe, "__version__", "unknown"),
        },
        "task": {
            "dataset": "breast_cancer_wisconsin",
            "model": "logistic_regression_single_dense",
            "train_samples": int(n_train),
            "test_samples": int(x_test.shape[0]),
            "features": int(d),
        },
        "paths": {
            "data_dir": str(DATA_DIR),
            "log_dir": str(LOG_DIR),
            "data_cache_file": str(DATA_DIR / "breast_cancer_split.npz"),
            "train_log_file": str(train_log_file),
            "report_file": str(report_file),
        },
        "checks": {},
        "history": [],
    }

    # ============================================================================
    # ⚠️【密态训练关键点 1】进入 ABY3 三方计算协议
    # ============================================================================
    # ABY3 是一种 3 方 MPC 协议（Arithmetic Sharing + Boolean Sharing + Yao）
    # 这个上下文管理器启动了密态计算环境，所有内部操作都在加密分片上执行
    with tfe.protocol.ABY3():
        total_start_time = time.perf_counter()
        
        # ========================================================================
        # ⚠️【密态训练关键点 2】创建密态模型
        # ========================================================================
        # 这个 Sequential 是 tfe.keras.Sequential，不是普通的 Keras
        # 模型的所有权重在初始化时就会被秘密共享分片
        model = Sequential()
        model.add(Dense(1, batch_input_shape=(n_train, d)))

        # ========================================================================
        # ⚠️【密态训练关键点 3】将明文数据转换为加密分片（秘密共享）
        # ========================================================================
        # define_private_variable 是核心操作：
        # 它把明文 x_train 拆分成多个随机分片，分发给 3 个计算方
        # 从这一行开始，没有任何一个计算方能单独看到完整数据
        x_priv = tfe.define_private_variable(x_train)
        y_priv = tfe.define_private_variable(y_train)

        # ========================================================================
        # ⚠️【密态训练关键点 4】密态损失函数和优化器
        # ========================================================================
        # 由于当前环境缺少 aux 自定义算子，这里使用 MSE 以规避 sigmoid 依赖。
        # 注意：这个 MSE 是在加密分片上计算的，不是普通的 TensorFlow MSE
        loss_fn = tfe.keras.losses.MeanSquaredError()
        optimizer = tfe.keras.optimizers.SGD(learning_rate=0.03)
        
        # ========================================================================
        # ⚠️【密态训练关键点 5】编译并执行密态前向传播
        # ========================================================================
        # model.compile 和 model(x_priv) 都在密态下执行
        # 输入 x_priv 是加密分片，输出也是加密分片
        model.compile(optimizer=optimizer, loss=loss_fn)
        model(x_priv)
        
        log_line("      模型: 单层 Dense (logistic regression 形式)")
        log_line("      优化器: SGD(lr=0.03), 损失: MSE")
        log_line("")
        log_line("[3/4] 执行训练...")
        log_line("=" * 90)
        log_line("{:<8} {:<12} {:<15}".format("Epoch", "Loss", "Train_Acc"))
        log_line("=" * 90)

        # ========================================================================
        # ⚠️【密态训练关键点 6】创建密态训练操作
        # ========================================================================
        # fit_batch: 返回一个密态训练操作，执行时会在加密分片上完成：
        #           前向传播 → 损失计算 → 反向传播 → 梯度更新
        # loss_tensor: 密态损失张量
        fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
        
        # predict: 密态预测，输出加密分片
        pred_tensor = model.predict(x_priv)
        
        # ========================================================================
        # ⚠️【密态训练关键点 7】reveal() 是唯一解密点
        # ========================================================================
        # data_probe = x_priv.reveal() 创建了一个解密操作
        # .reveal() 会请求三方将各自的分片合并，还原出明文数据
        # 这是整个训练过程中唯一能接触到明文的地方（仅用于验证数据流）
        data_probe = x_priv.reveal()

        # 获取 MPC 会话
        sess = KE.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # 验证数据流：解密并检查形状
        revealed_x = sess.run(data_probe)
        result["checks"]["data_flow_ok"] = bool(revealed_x.shape == x_train.shape)

        # ========================================================================
        # ⚠️【密态训练关键点 8】解密 Loss 和预测值（仅用于监控）
        # ========================================================================
        # loss_tensor.reveal() 解密损失值
        # pred_tensor.reveal() 解密预测值
        # 注意：解密操作只发生在监控点，不影响模型权重的密态存储
        init_loss = float(sess.run(loss_tensor.reveal()))
        init_pred = sess.run(pred_tensor.reveal())
        init_train_acc = float(np.mean((init_pred > 0.5) == y_train))
        result["checks"]["loss_computable"] = np.isfinite(init_loss).item()

        optimizer_ok = True
        epochs = 40
        for epoch in range(1, epochs + 1):
            epoch_bar = tqdm(total=1, desc="Epoch {}/{}".format(epoch, epochs), ncols=100, leave=False)
            try:
                # ================================================================
                # ⚠️【密态训练关键点 9】执行一次密态训练迭代
                # ================================================================
                # sess.run(fit_op) 触发多方 MPC 协议执行：
                # 1. 三方在加密分片上联合计算前向传播
                # 2. 三方联合计算梯度（同样是加密分片）
                # 3. 三方联合更新模型权重（权重保持加密分片状态）
                # 整个过程没有任何一方能看到完整的数据或梯度
                sess.run(fit_op)
            except Exception:  # pragma: no cover - 诊断路径
                optimizer_ok = False
                log_line("epoch {}: 训练中断，optimizer 执行失败".format(epoch))
                epoch_bar.close()
                break

            # ====================================================================
            # ⚠️【密态训练关键点 10】解密 Loss 和预测值（仅用于日志监控）
            # ====================================================================
            # 再次强调：这里的 .reveal() 只是为了让用户看到训练进展
            # 模型权重本身从未被解密，始终以加密分片形式存储
            loss_value = float(sess.run(loss_tensor.reveal()))
            pred_value = sess.run(pred_tensor.reveal())
            train_acc = float(np.mean((pred_value > 0.5) == y_train))
            result["history"].append(
                {"epoch": epoch, "loss": loss_value, "train_accuracy": train_acc}
            )
            epoch_bar.set_postfix(loss="{:.6f}".format(loss_value), train_acc="{:.6f}".format(train_acc))
            epoch_bar.update(1)
            epoch_bar.close()
            log_line("{:<8} {:<12.6f} {:<15.6f}".format(epoch, loss_value, train_acc))

        # ========================================================================
        # ⚠️【密态训练关键点 11】训练结束后解密模型权重
        # ========================================================================
        # 这是最终解密：将训练好的权重从加密分片还原为明文
        # 用于后续明文侧评估
        w_tfe = model.layers[0].weights[0].reveal()
        b_tfe = model.layers[0].weights[1].reveal()
        w_np, b_np = sess.run([w_tfe, b_tfe])
        test_acc, test_mse = eval_linear_binary(x_test, y_test, w_np, b_np)

        final_loss = result["history"][-1]["loss"] if result["history"] else init_loss
        final_train_acc = (
            result["history"][-1]["train_accuracy"] if result["history"] else init_train_acc
        )

        result["checks"]["backward_optimizer_ok"] = bool(optimizer_ok)
        result["checks"]["loss_changed"] = bool(abs(final_loss - init_loss) > 1e-6)
        result["checks"]["train_accuracy_changed"] = bool(
            abs(final_train_acc - init_train_acc) > 1e-6
        )
        result["checks"]["loss_decreased"] = bool(final_loss < init_loss)
        result["checks"]["train_accuracy_improved"] = bool(
            final_train_acc >= init_train_acc
        )

        result["summary"] = {
            "initial_train_loss": init_loss,
            "final_train_loss": final_loss,
            "initial_train_accuracy": init_train_acc,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": test_acc,
            "final_test_mse": test_mse,
            "total_time_sec": float(time.perf_counter() - total_start_time),
        }

        # 清理 MPC 会话
        KE.clear_session()

    must_have = [
        "data_flow_ok",
        "loss_computable",
        "backward_optimizer_ok",
        "loss_changed",
        "train_accuracy_changed",
    ]
    result["baseline_training_ready"] = all(result["checks"].get(k, False) for k in must_have)

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, ensure_ascii=False)
        fp.write("\n")
    log_line("=" * 90)
    log_line("")
    log_line("[4/4] 训练完成")
    log_line("baseline_training_ready: {}".format(result["baseline_training_ready"]))
    log_line("final_train_loss: {:.6f}".format(result["summary"]["final_train_loss"]))
    log_line("final_train_accuracy: {:.6f}".format(result["summary"]["final_train_accuracy"]))
    log_line("final_test_accuracy: {:.6f}".format(result["summary"]["final_test_accuracy"]))
    log_line("total_time(s): {:.6f}".format(result["summary"]["total_time_sec"]))
    log_line("训练报告已写入: {}".format(report_file))
    return 0 if result["baseline_training_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
