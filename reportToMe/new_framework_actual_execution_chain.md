# 纯代码事实执行链

只描述当前已实现边界；未运行任何代码。

## A. 最新正Stage-B C5（正式TFE；未接Stage C）

```text
[PRECOMPUTED] Official MTT ipc=50 synthetic images + integer labels
→ [LOCAL/OFFLINE] one shared NewMTTConvNet per dataset/seed
→ [PRECOMPUTED] test-free cached 2048-D features (48/class)
→ [LOCAL] host uses public labels to split 40/class local + 8/class global anchor
→ [LOCAL] host partitions local pool into five logical clients
→ [SECRET-SHARE] client features and host one-hot labels become TFE private variables
→ [MPC: TFE/ABY3 single-machine three-server simulation]
   five linear heads train independently with full-batch softmax CE
→ [MPC] uniform average of classifier W,b
→ [SECRET-SHARE] pre-balanced global anchor features/one-hot labels
→ [MPC] C5 initializes directly from shared average and performs 40 Anchor-CE updates
→ [REVEALED] final C1/C4/C5 W,b
→ [LOCAL] save plaintext NPZ and evaluate development correctness
→ [STOP] no Stage C, DP-SGD, End-to-End, official test, or locked holdout
```

没有client-specific D̃i/Ai，没有secure class balancing，也没有plaintext θi上传。

## B. 当前True Ours Figure-2正式链

此链使用旧`secure_average_only` initializer，不是最新C5。

```text
[PRECOMPUTED] Official-MTT cached 2048-D features
→ [LOCAL] host makes balanced five-client logical partitions/batches
→ [SECRET-SHARE] batch features/one-hot labels enter through logical server0;
   initial linear W,b become private variables
→ [MPC] five local linear heads train with logits-MSE;
   optional periodic secure FedAvg follows frozen config
→ [MPC] public uniform/sample-count coefficients form NoKD average W,b
→ [REVEALED] Secure-Average-only initializer is saved as ordinary NPZ;
   aggregate development logits are revealed at checkpoint epochs

→ [LOCAL/PRECOMPUTED] a separate Stage-C process reads the revealed initializer
   and complete 48/class Official-MTT Stage-C feature cache
→ [SECRET-SHARE] initializer becomes new private W,b; no Stage-B share is reused
→ [LOCAL] host os.urandom performs independent per-record Poisson selection;
   host knows IDs and builds a padded batch and mask
→ [SECRET-SHARE] padded features, host one-hot labels, and mask enter via server0
→ [MPC] linear logits/errors, analytical per-example norms, clip factors,
   and aggregated clipped W/b gradients
→ [LOCAL ON EACH LOGICAL SERVER] server0/1/2 independently generate
   checkpointed CSPRNG Gaussian parts N(0,1/3)
→ [SECRET-SHARE + MPC] private-input and sum the three parts under ABY3;
   add scaled noise to aggregated clipped gradient
→ [MPC] update W,b; registered adaptive steps update private Ct from a
   DP-noised below-bound fraction
→ [REVEALED DIAGNOSTICS] Ct every step; noisy fraction/proposed Ct at adaptive
   steps; aggregate development logits/loss at log intervals
→ [REVEALED FINAL] final W,b,Ct saved as ordinary NPZ
→ [PRIVACY SCOPE] epsilon covers only conditional Stage-C sampled-gradient
   and adaptive-count mechanisms, not Stage A/B or the end-to-end model
```

## C. 当前不能画成已实现的链

```text
private client raw data
→ per-client local MTT
→ client-specific Dtilde_i / A_i
→ secure class balancing
→ plaintext local theta_i upload
→ latest C5
→ same-share seamless Stage C
→ end-to-end record-level DP model
```

缺失边分别是：client-local MTT、client anchor贡献、secure balancing、plaintext θi上传
语义、latest C5→Stage C、share复用和端到端DP。
