#!/usr/bin/env python3
"""
Minimal end-to-end training validation on TF Encrypted (tfe 0.8.x style).

Validates 4 key points:
1) data enters TF Encrypted flow
2) loss is computable
3) backward + optimizer step runs
4) loss / accuracy changes during training
"""

import json
import platform
import sys

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.keras import Sequential
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.layers import Dense


def build_toy_binary_dataset(num_samples=256, seed=7):
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, 2).astype(np.float32)
    w_true = np.array([1.5, -2.0], dtype=np.float32)
    logits = x.dot(w_true) + 0.1 * rng.randn(num_samples).astype(np.float32)
    y = (logits > 0).astype(np.float32).reshape(-1, 1)
    return x, y


def main():
    np.random.seed(7)
    tf.set_random_seed(7)
    tf.reset_default_graph()

    x_np, y_np = build_toy_binary_dataset()
    n, d = x_np.shape

    result = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tf_encrypted": getattr(tfe, "__version__", "unknown"),
        },
        "task": {
            "dataset": "synthetic_tabular_binary",
            "model": "logistic_regression",
            "samples": int(n),
            "features": int(d),
        },
        "checks": {},
        "history": [],
    }

    with tfe.protocol.ABY3():
        model = Sequential()
        model.add(Dense(1, batch_input_shape=(n, d)))

        x_priv = tfe.define_private_variable(x_np)
        y_priv = tfe.define_private_variable(y_np)

        # This confirms that data has entered the TFE private-tensor pipeline.
        data_probe = x_priv.reveal()

        # Use MSE here to avoid sigmoid-dependent ops that require aux custom op.
        # This still validates data flow, loss, backward, optimizer, and metric movement.
        loss_fn = tfe.keras.losses.MeanSquaredError()
        optimizer = tfe.keras.optimizers.SGD(learning_rate=0.05)
        model.compile(optimizer=optimizer, loss=loss_fn)
        model(x_priv)

        fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
        logits_tensor = model.predict(x_priv)

        sess = KE.get_session()
        sess.run(tf.global_variables_initializer())

        revealed_x = sess.run(data_probe)
        result["checks"]["data_flow_ok"] = bool(revealed_x.shape == x_np.shape)

        initial_loss = float(sess.run(loss_tensor.reveal()))
        initial_logits = sess.run(logits_tensor.reveal())
        initial_acc = float(np.mean((initial_logits > 0.5) == y_np))
        result["checks"]["loss_computable"] = np.isfinite(initial_loss).item()

        epochs = 30
        optimizer_ran = True
        for epoch in range(1, epochs + 1):
            try:
                sess.run(fit_op)
            except Exception:  # pragma: no cover - diagnostics path
                optimizer_ran = False
                break

            loss_value = float(sess.run(loss_tensor.reveal()))
            logits_value = sess.run(logits_tensor.reveal())
            acc_value = float(np.mean((logits_value > 0.5) == y_np))
            score_mean = float(np.mean(logits_value))

            result["history"].append(
                {
                    "epoch": epoch,
                    "loss": loss_value,
                    "accuracy": acc_value,
                    "score_mean": score_mean,
                }
            )

        final_loss = result["history"][-1]["loss"] if result["history"] else initial_loss
        final_acc = result["history"][-1]["accuracy"] if result["history"] else initial_acc

        result["checks"]["backward_optimizer_ok"] = bool(optimizer_ran)
        result["checks"]["loss_changed"] = bool(abs(final_loss - initial_loss) > 1e-6)
        result["checks"]["accuracy_changed"] = bool(abs(final_acc - initial_acc) > 1e-6)
        result["checks"]["loss_decreased"] = bool(final_loss < initial_loss)
        result["checks"]["accuracy_improved"] = bool(final_acc >= initial_acc)

        result["summary"] = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "initial_accuracy": initial_acc,
            "final_accuracy": final_acc,
        }

        KE.clear_session()

    required = [
        "data_flow_ok",
        "loss_computable",
        "backward_optimizer_ok",
        "loss_changed",
        "accuracy_changed",
    ]
    result["baseline_training_ready"] = all(result["checks"].get(k, False) for k in required)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["baseline_training_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
