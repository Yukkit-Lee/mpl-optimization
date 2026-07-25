# Cross-Stage Pipeline

## Algorithmic Main Pipeline

```text
Official MTT Distilled Images
-> Local PyTorch Full ConvNets
-> Full-ConvNet KD
-> KD-Avg / KD-ReferenceWeighted / KD-Best
-> Heldout-Reference-Guided Chosen Full ConvNet
-> Same PyTorch Full ConvNet Plain / Fixed-DP / Adaptive-DP
```

- Backend: `local_plaintext_pytorch`.
- No 2048-D feature cache is used by the current entry points.
- Stage B checkpoints contain the complete ConvNet state dictionary.
- Checkpoint SHA256 and deterministic state signatures are verified before Stage C.
- The held-out reference set is sampled from the real training split and is a
  private data dependency. It is disjoint from distilled/local data and test data.
- Test data is used only for fixed-interval readonly reporting and final metrics.

## Secure-System Validation

```text
Official MTT Distilled Images
-> Flatten
-> TFE Lightweight Dense Model
-> ABY3 Secret-Shared Training
```

- This representative system experiment does not replace the full-ConvNet method.
- It records every revealed value and never silently falls back to plaintext.
- Exact secure per-example clipping, adaptive clipping, and MPC noise generation
  remain blocked in the installed TFE stack.

## Privacy

- Scope: `stage_c_only`.
- Accountant: Opacus RDP for sampled Gaussian Fixed-DP and Adaptive-DP.
- Stage A and Stage B are data-dependent and not DP-protected.
- Binomial simulation is not assigned a Gaussian epsilon.

> The reported privacy budget covers Stage C only. Stages A and B are
> data-dependent and are not included in the current record-level DP accounting;
> therefore, the reported epsilon does not constitute end-to-end DP for the
> original local datasets.

## Safe Commands

All commands below enforce one experiment, one visible GPU, and one CPU/BLAS thread.

```bash
cd /mnt/code/lixujie/tfe_experiments

# Full-ConvNet Stage B
bash scripts/run_safe_experiment.sh \
  /home/lixujie/.conda/envs/distillation/bin/python \
  exp_module_b_kd_noniid_official.py --dataset cifar10

# Full-ConvNet A+B+C
bash scripts/run_safe_experiment.sh \
  /home/lixujie/.conda/envs/distillation/bin/python \
  exp_end2end_framework.py --dataset cifar10

# TFE lightweight system validation
CUDA_VISIBLE_DEVICES="" bash scripts/run_safe_experiment.sh \
  /home/lixujie/.conda/envs/tfe/bin/python \
  exp_stageC_tfe_secure_system.py --dataset cifar10
```

Use `--dataset cifar100` or `--dataset tiny` for the other algorithmic datasets.
Use `--quick` only for structural smoke tests.
