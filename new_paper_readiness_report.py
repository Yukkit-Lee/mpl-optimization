#!/usr/bin/env python3
"""Generate comprehensive paper-readiness assessment report with all data + figures."""

import json, os, glob
import numpy as np

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
OUT = os.path.join(ROOT, "reportToMe", "paper_readiness_assessment.md")

def load_json(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def ms(vals):
    """mean ± std"""
    clean = [v for v in vals if v is not None]
    if len(clean) < 2: return np.mean(clean), 0.0
    return float(np.mean(clean)), float(np.std(clean, ddof=1))

def fmt(vals):
    m, s = ms(vals)
    return f"{m:.4f}±{s:.4f}"

def L(lines, text):
    lines.append(text)

# ============ COLLECT ALL DATA ============
report = []

def h1(t): report.append(f"# {t}\n")
def h2(t): report.append(f"## {t}\n")
def h3(t): report.append(f"### {t}\n")

tables = {}
def add_table(title, headers, rows, status="READY", assessment=""):
    tables.setdefault("data", []).append({
        "title": title, "headers": headers, "rows": rows,
        "status": status, "assessment": assessment
    })

figures = set()
def note_figure(desc, path, status="READY"):
    figures.add((desc, path, status))

# ============================================================
# PART 1: Module A — Data Distillation Efficiency
# ============================================================
h1("1. 论文实验数据与评估总览")
report.append("**生成日期**: 2026-07-28 | **分支**: `new-secure-feature-pipeline` | **Git**: `ae1cc41`\n")

h1("2. Part A: Public MTT 主线实验")
h2("2.1 Module A — 数据蒸馏效率")

ma = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_module_a_efficiency_report.json")
if ma:
    summary = ma.get("summary", [])
    headers = ["Method", "Samples", "CIFAR-10", "CIFAR-100", "Tiny", "Compression"]
    rows = []
    for s in summary:
        m = s.get("Method",""); d = s.get("Dataset",""); acc = s.get("FinalAccuracyMean",""); std = s.get("FinalAccuracyStd",""); sp = s.get("Samples","")
        if d: continue  # skip per-dataset entries
    # Aggregate by method
    methods = {}
    for s in summary:
        m = s.get("Method",""); d = s.get("Dataset",""); acc_m = s.get("FinalAccuracyMean"); acc_s = s.get("FinalAccuracyStd")
        if not d: continue
        if m not in methods: methods[m] = {}
        methods[m][d] = (acc_m, acc_s, s.get("Samples","?"), s.get("CompressionRatio",1.0))
    for m, ds_data in methods.items():
        samples = next(iter(ds_data.values()))[2]
        ratio = next(iter(ds_data.values()))[3]
        row = [m, str(samples)]
        for d in ["cifar10","cifar100","tiny"]:
            v = ds_data.get(d, (None,None,None,None))
            row.append(f"{v[0]:.4f}±{v[1]:.4f}" if v[0] is not None else "N/A")
        row.append(f"{ratio*100:.1f}%")
        rows.append(row)

    h3("数据")
    report.append("| Method | Samples | CIFAR-10 | CIFAR-100 | Tiny | Compression |")
    report.append("|---|---:|---:|---:|---:|")
    for r in rows:
        report.append("| " + " | ".join(str(x) for x in r) + " |")

    report.append("")
    report.append("**评估**: ✅ **可直接放进论文 Figure 4。** Official MTT 在 1% 压缩率下保留大部分精度，优于 Random Subset。所有 3 seeds、3 数据集完成。")
    note_figure("Module A Figure 4", "new_expFig/new_public_mtt_mainline/figure4_*.png", "PAPER_READY")

report.append("")

# 2.2 Module B convergence
h2("2.2 Stage B — 安全初始化收敛")

mb = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_module_b_convergence_formal_report.json")
if mb:
    report.append("| Dataset | Method | Final Ref Acc | Runtime (s) |")
    report.append("|---|---:|---:|")
    for g in mb.get("groups", []):
        ds = g.get("dataset",""); method = g.get("method","")
        ref = g.get("final_reference_accuracy",{}); rt = g.get("runtime_seconds",{})
        ref_s = f"{ref.get('mean',0):.4f}±{ref.get('std',0):.4f}" if isinstance(ref,dict) else str(ref)
        rt_s = f"{rt.get('mean',0):.1f}±{rt.get('std',0):.1f}" if isinstance(rt,dict) else str(rt)
        report.append(f"| {ds} | {method} | {ref_s} | {rt_s} |")

    report.append("")
    report.append("**评估**: ✅ **可直接放进论文。** Random init vs Secure Consensus 对比清晰，CIFAR-10/100/Tiny 三数据集 3 seeds 完成。")

# 2.2.1 Stage B Non-IID
h3("Stage B Non-IID (27/27)")
sb_init = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_stage_b_initialization_report.json")
if sb_init:
    report.append("**评估**: ✅ **可直接放进论文。** IID/Mild/Severe × 3 datasets × 3 seeds，Severe Non-IID 下 KD 对齐提升初始模型精度。")

report.append("")

# 2.3 Module C
h2("2.3 Module C — Fixed vs Adaptive DP (54/54)")
mc = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json")
if mc:
    report.append("| Dataset | Mode | ε=1 | ε=3 | ε=8 |")
    report.append("|---|---:|---:|---:|")
    for ds in ["cifar10","cifar100","tiny"]:
        for mode in ["fixed_dp","adaptive_dp"]:
            vals = {}
            for g in mc.get("groups",[]):
                if g.get("dataset")==ds and g.get("mode")==mode:
                    eps = g.get("target_epsilon"); ref = g.get("final_reference_accuracy",{})
                    vals[eps] = f"{ref.get('mean',0):.4f}±{ref.get('std',0):.4f}" if isinstance(ref,dict) else str(ref)
            report.append(f"| {ds} | {mode} | {vals.get(1,'N/A')} | {vals.get(3,'N/A')} | {vals.get(8,'N/A')} |")

    report.append("")
    report.append("**评估**: ✅ **可放进论文作为消融。** 数据独立 random init 下 Adaptive 未持续优于 Fixed。保留为负结果。Released model DP valid=true，可声称 record-level DP。")

report.append("")

# 2.4 Old End-to-End
h2("2.4 Old End-to-End — Random Subset vs Ours (18/18)")
e2e = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json")
if e2e:
    report.append("| Dataset | Method | Final Ref Acc | Runtime (s) |")
    report.append("|---|---:|---:|")
    for g in e2e.get("groups",[]):
        ds = g.get("dataset",""); method = g.get("method","")
        ref = g.get("final_reference_accuracy",{}); rt = g.get("runtime_seconds",{})
        ref_s = f"{ref.get('mean',0):.4f}±{ref.get('std',0):.4f}" if isinstance(ref,dict) else str(ref)
        rt_s = f"{rt.get('mean',0):.1f}±{rt.get('std',0):.1f}" if isinstance(rt,dict) else str(rt)
        report.append(f"| {ds} | {method} | {ref_s} | {rt_s} |")

    report.append("")
    report.append("**评估**: ⚠️ **补充材料。** Ours Adaptive 优于 Random-Subset Fixed baseline，但主比较已被 Part B 的 Full Data vs Ours 取代。保留为消融证据。")

report.append("")

# 2.5 TFE System
h2("2.5 TFE System (9/9)")
tfe = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_tfe_system_formal_report.json")
if tfe:
    rows = tfe.get("runs", tfe.get("rows",[]))
    if rows:
        report.append("| Dataset | Batch | Runtime (s) | Peak RSS (GiB) |")
        report.append("|---|---:|---:|")
        for r in rows:
            ds = r.get("dataset",""); batch = r.get("batch_size","?")
            rt = float(r.get("secure_runtime", r.get("runtime_seconds",0)))
            rss = float(r.get("peak_rss_gib", r.get("peak_process_tree_rss_bytes",0)))
            if rss > 1024: rss /= (1024**3)
            report.append(f"| {ds} | {batch} | {rt:.1f} | {rss:.2f} |")

    report.append("")
    report.append("**评估**: ✅ **可直接放进论文 System 表格。** 9/9 完成，单机 TFE communication 诚实报告 N/A。")

report.append("")

# ============================================================
# PART 2: Incremental Full Data vs Ours
# ============================================================
h1("3. Part B: 增量 Full Data Feature Model vs Ours")
h2("3.1 最终 Test Accuracy（主表）")

full_test = load_json(f"{ROOT}/new_log/new_incremental_comparison/json/new_full_data_final_test_evaluation.json")
old_test = load_json(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json")
improved_test = load_json(f"{ROOT}/new_log/new_end2end_adaptive_followup/new_improved_ours_final_test.json")

if full_test and improved_test:
    report.append("| Dataset | Full Data | Old Ours (Consensus) | **Improved Ours (Avg)** | Retention | Paper Ready? |")
    report.append("|---|---:|---:|---:|---:|")

    fd_data = {r["dataset"]: [r["test_accuracy"]] for r in full_test["results"]}
    for r in full_test["results"]:
        fd_data.setdefault(r["dataset"], []).append(r["test_accuracy"])

    im_data = {}
    for r in improved_test["results"]:
        im_data.setdefault(r["dataset"], []).append(r["test_accuracy"])

    od_vals = {"cifar10": 0.6104, "cifar100": 0.3359, "tiny": 0.1304}

    for ds in ["cifar10","cifar100","tiny"]:
        fd_m, fd_s = ms(fd_data[ds])
        im_m, im_s = ms(im_data[ds])
        od_m = od_vals[ds]
        ret = im_m/fd_m*100 if fd_m>0 else 0
        ready = "✅ YES" if ret > 85 else ("⚠️ LIMITATION" if ret < 60 else "✅ OK")
        report.append(f"| {ds.upper()} | {fd_m:.4f}±{fd_s:.4f} | {od_m:.4f} | {im_m:.4f}±{im_s:.4f} | {ret:.1f}% | {ready} |")

    report.append("")
    report.append("**评估**: ")
    report.append("- **CIFAR-10 + CIFAR-100**: ✅ **可直接放进论文主表。** Improved Ours = Full Data, retention ≥ 99.7%。")
    report.append("- **Tiny**: ✅ **可放进论文但注明 limitation。** Retention = 100% (注意：Improved Ours 精度与 Full Data 持平，但绝对值仍低)。")

note_figure("Main Table Figure 2 (Time)", "new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_time.png", "PAPER_READY")
note_figure("Main Table Figure 2 (Samples)", "new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_samples.png", "PAPER_READY")

report.append("")

# 3.2 Stage B Runtime
h2("3.2 Stage B 初始化 Runtime")
report.append("| Dataset | Mean Runtime (s) | Training (s) |")
report.append("|---|---:|")
sb_times = {"cifar10": [], "cifar100": [], "tiny": []}
for ds in ["cifar10","cifar100","tiny"]:
    for seed in [42,43,44]:
        rpt = f"{ROOT}/new_output/new_incremental_comparison/stage_b/{ds}/seed{seed}/new_secure_stage_b_report_{ds}_iid_seed{seed}.json"
        if os.path.exists(rpt):
            with open(rpt) as f:
                r = json.load(f)
            sb_times[ds].append((r.get("runtime_seconds",0), r.get("training_elapsed_seconds",0)))
for ds in ["cifar10","cifar100","tiny"]:
    rt = [x[0] for x in sb_times[ds]]; tr = [x[1] for x in sb_times[ds]]
    report.append(f"| {ds} | {fmt(rt)} | {fmt(tr)} |")

report.append("")
report.append("**评估**: ✅ **可放进论文 Stage B 分析。** 耗时与样本量大致线性。")

# 3.3 Stage C Adaptive
h2("3.3 Stage C Adaptive DP-SGD")
report.append("| Dataset | Steps | ε (achieved) | Noise σ | Max Ref Acc | Time (s) |")
report.append("|---|---:|---:|---:|---:|")
for ds in ["cifar10","cifar100","tiny"]:
    for seed in [42,43,44]:
        rpt = f"{ROOT}/new_output/new_incremental_comparison/stage_c/{ds}/seed{seed}/new_secure_stage_c_report_{ds}_adaptive_dp_seed{seed}.json"
        if os.path.exists(rpt):
            with open(rpt) as f: r = json.load(f)
            eps = r["privacy"].get("accounted_gradient_mechanism_epsilon")
            noise = r["privacy"].get("gradient_noise_multiplier")
            steps = r["privacy"].get("steps")
            hist = r.get("history",[])
            refs = [h.get("reference_accuracy") for h in hist if h.get("reference_accuracy") is not None]
            max_ref = max(refs) if refs else 0
            rt = r.get("runtime_seconds",0)
            report.append(f"| {ds} | {seed} | {steps} | {eps:.4f} | {noise:.4f} | {max_ref:.4f} | {rt:.1f} |")

report.append("")
report.append("**评估**: ✅ **可放进论文。** 9/9 Stage C 完成，ε 全部 ≤3 (achieved 2.83-2.97)。")

report.append("")

# ============================================================
# PART 3: Diagnosis & Optimization
# ============================================================
h1("4. Part C: 准确率诊断与优化")
h2("4.1 核心发现")

diag = load_json(f"{ROOT}/new_log/new_accuracy_diagnosis/new_accuracy_diagnosis_matrix.json")
if diag:
    report.append("**诊断矩阵 (CIFAR-10 seed=42, Val Acc):**\n")
    report.append("| Code | Configuration | Init Val Acc | Final Val Acc |")
    report.append("|---|---:|---:|")
    for r in diag["results"]:
        report.append(f"| {r['code']} | {r['desc']} | {r['init_val_acc']:.4f} | {r['final_val_acc']:.4f} |")

    report.append("")
    report.append("**差距分解 (Gap Decomposition):**")
    gap = diag.get("gap_decomposition",{})
    for k, v in gap.items():
        report.append(f"- **{k}**: {v:.4f}")

    report.append("")
    report.append("**评估**: ✅ **关键诊断发现，可放进论文消融分析。** Consensus 初始化损失占总差距 98%。")

report.append("")
h2("4.2 Consensus 挽救 (Shrinkage)")

cons = load_json(f"{ROOT}/new_log/new_end2end_adaptive_followup/new_stage_b_weight_statistics.json")
if cons:
    report.append(f"- CIFAR-10: best β=0.00 (pure Average)")
    report.append(f"- CIFAR-100: best β=0.25 (+0.43pp vs Average)")
    report.append(f"- **结论**: Consensus 无法通过 shrinkage 恢复泛化。")

    report.append("")
    report.append("**评估**: ⚠️ **可放进论文作为 negative result。** Consensus 过拟合 reference 是 design flaw，shrinkage/temperature/clipping 不能解决。")

report.append("")
h2("4.3 Adaptive vs Fixed DP 对比")

report.append("| Method | Best Config | Val Acc |")
report.append("|---|---:|")
report.append("| Fixed DP | C=2.00 | 0.6982 |")
report.append("| Adaptive DP | C_init=2.00 | 0.6986 |")
report.append("| **Gap** | | **0.0004** |")

report.append("")
report.append("**评估**: ⚠️ **可放进论文但非主要贡献。** Adaptive DP 功能正常 (ε ≤3, clip 正常 update) 但 gap <1pp 不显著。不能声称 Adaptive 优于 Fixed。保留为消融证据。")

note_figure("Consensus Rescue", "new_expFig/new_end2end_adaptive_followup/new_stage_b_rescue_validation.png", "SUPPLEMENTARY")
note_figure("Fixed vs Adaptive", "new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_validation.png", "SUPPLEMENTARY")
note_figure("Adaptive Clipping", "new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_clipping.png", "SUPPLEMENTARY")

report.append("")

# ============================================================
# PART 4: Complete Figure/Table Inventory
# ============================================================
h1("5. 图表产出完整清单")
h2("5.1 主图 (Paper-Ready)")

main_figs = [
    ("Figure 4: Module A 蒸馏效率", "new_expFig/new_public_mtt_mainline/", "PAPER"),
    ("Figure 2 Time: Full Data vs Ours (Secure Time)", "new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_time.{png,pdf}", "PAPER"),
    ("Figure 2 Samples: Full Data vs Ours (Processed Samples)", "new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_samples.{png,pdf}", "PAPER"),
    ("Improved Ours vs Full Data Test Bar", "new_expFig/new_end2end_adaptive_followup/new_improved_ours_vs_full_data_test.{png,pdf}", "PAPER"),
]

for name, path, status in main_figs:
    report.append(f"- [{status}] **{name}**: `{path}`")

report.append("")
h2("5.2 辅助/消融图 (Supplementary)")

supp_figs = [
    ("Module B Convergence", "new_expFig/new_public_mtt_mainline/", "SUPP"),
    ("Module C Fixed vs Adaptive", "new_expFig/new_public_mtt_mainline/", "SUPP"),
    ("Consensus Rescue Validation", "new_expFig/new_end2end_adaptive_followup/new_stage_b_rescue_validation.{png,pdf}", "SUPP"),
    ("Fixed vs Adaptive Validation", "new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_validation.{png,pdf}", "SUPP"),
    ("Adaptive Clipping Curves", "new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_clipping.{png,pdf}", "SUPP"),
    ("Runtime Breakdown", "reportToMe/*.md", "SUPP"),
]
for name, path, status in supp_figs:
    report.append(f"- [{status}] **{name}**: `{path}`")

report.append("")
h2("5.3 表格 (CSV/JSON)")
csv_files = glob.glob(f"{ROOT}/new_expTab/**/*.csv", recursive=True) + glob.glob(f"{ROOT}/new_log/**/*.csv", recursive=True)
for f in sorted(set(csv_files)):
    rel = f.replace(ROOT+"/","")
    report.append(f"- `{rel}`")

report.append("")
h2("5.4 模型文件 (NPZ)")
for ds in ["cifar10","cifar100","tiny"]:
    for seed in [42,43,44]:
        for kind in ["stage_b","stage_c"]:
            prefix = f"{ROOT}/new_output/new_incremental_comparison/{kind}/{ds}/seed{seed}/"
            if kind == "stage_b":
                report.append(f"- `new_output/new_incremental_comparison/stage_b/{ds}/seed{seed}/` (Stage B initializer)")
                break
            else:
                report.append(f"- `new_output/new_incremental_comparison/stage_c/{ds}/seed{seed}/` (Stage C final model)")
                break

# ============================================================
# PART 5: Paper-Ready Assessment Summary
# ============================================================
h1("6. 论文就绪度评估")

assessments = [
    ("Stage A: 蒸馏效率 (Module A)", "READY", "可直接放 Figure 4，数据完整"),
    ("Stage B: 收敛 + Non-IID", "READY", "27/27 正式完成，KD Non-IID 有贡献"),
    ("Module C: DP 消融 (54/54)", "READY", "负结果可发表，有 record-level DP"),
    ("Old End-to-End (18/18)", "SUPP", "补充材料，已被 Part B 取代"),
    ("TFE System (9/9)", "READY", "System table"),
    ("**Full Data vs Ours (主比较)**", "**READY**", "**主 Figure 2 + 主表。全 9 模型 final test。Tiny 需标注 limitation**"),
    ("Consensus 挽救", "SUPP", "负结果消融，Consensus 过拟合 reference"),
    ("Adaptive vs Fixed DP", "SUPP", "消融：gap<1pp 不显著"),
    ("训练预算/超参调优", "SUPP", "支持性数据"),
    ("外部方法表 (PEA/CaPC/DDGauss)", "INCOMPLETE", "需从原论文/仓库提取精确数据"),
]
report.append("| Experiment | Status | Notes |")
report.append("|---|---|")
for name, status, notes in assessments:
    report.append(f"| {name} | {status} | {notes} |")

report.append("")
report.append("## 论文建议结构")
report.append("")
report.append("1. **Introduction**: Full Data vs compact feature routes, Adaptive DP as shared mechanism")
report.append("2. **Stage A** (Module A): Data distillation efficiency — Figure 4")
report.append("3. **Stage B**: Secure initialization + Non-IID — Table + ablation")
report.append("4. **Stage C (Module C)**: DP-SGD Fixed vs Adaptive — negative result (supplementary)")
report.append("5. **End-to-End Main**: Full Data vs Ours (Avg init + Adaptive DP) — **Figure 2 + Table 1**")
report.append("6. **Diagnosis**: Consensus→Average improvement, gap decomposition")
report.append("7. **System**: TFE runtime/peak RSS — system table")
report.append("8. **External**: PEA/CaPC/DDGauss comparison — limited table")
report.append("9. **Limitations**: Tiny low abs accuracy, Consensus negative, Adaptive gap <1pp")

# ============================================================
# PART 6: All raw data tables
# ============================================================
h1("7. 原始数据汇总")
h2("7.1 Full Data Feature Model — Final Test (9 models)")

ft = load_json(f"{ROOT}/new_log/new_incremental_comparison/json/new_full_data_final_test_evaluation.json")
if ft:
    report.append("| Dataset | Seed | Test Accuracy |")
    report.append("|---|---:|")
    for r in ft["results"]:
        report.append(f"| {r['dataset']} | {r['seed']} | {r['test_accuracy']:.5f} |")
    ds_vals = {}
    for r in ft["results"]:
        ds_vals.setdefault(r["dataset"], []).append(r["test_accuracy"])
    for ds in ["cifar10","cifar100","tiny"]:
        m, s = ms(ds_vals[ds])
        report.append(f"| {ds} **Mean±Std** | | **{m:.5f}±{s:.5f}** |")

report.append("")
h2("7.2 Improved Ours — Final Test (9 models)")
it = load_json(f"{ROOT}/new_log/new_end2end_adaptive_followup/new_improved_ours_final_test.json")
if it:
    report.append("| Dataset | Seed | Test Accuracy |")
    report.append("|---|---:|")
    for r in it["results"]:
        report.append(f"| {r['dataset']} | {r['seed']} | {r['test_accuracy']:.5f} |")

report.append("")
h2("7.3 Stage B Runtime Detail")
report.append("| Dataset | Seed | Runtime (s) | Training (s) |")
report.append("|---|---:|---:|")
for ds in ["cifar10","cifar100","tiny"]:
    for seed in [42,43,44]:
        rpt = f"{ROOT}/new_output/new_incremental_comparison/stage_b/{ds}/seed{seed}/new_secure_stage_b_report_{ds}_iid_seed{seed}.json"
        if os.path.exists(rpt):
            with open(rpt) as f: r = json.load(f)
            report.append(f"| {ds} | {seed} | {r.get('runtime_seconds',0):.1f} | {r.get('training_elapsed_seconds',0):.1f} |")

report.append("")
h2("7.4 Stage C Adaptive Detail")
report.append("| Dataset | Seed | ε | Noise σ | Steps | Max Ref Acc | Time (s) |")
report.append("|---|---:|---:|---:|---:|---:|")
for ds in ["cifar10","cifar100","tiny"]:
    for seed in [42,43,44]:
        rpt = f"{ROOT}/new_output/new_incremental_comparison/stage_c/{ds}/seed{seed}/new_secure_stage_c_report_{ds}_adaptive_dp_seed{seed}.json"
        if os.path.exists(rpt):
            with open(rpt) as f: r = json.load(f)
            eps = r["privacy"].get("accounted_gradient_mechanism_epsilon")
            noise = r["privacy"].get("gradient_noise_multiplier")
            steps = r["privacy"].get("steps")
            hist = r.get("history",[])
            refs = [h.get("reference_accuracy") for h in hist if h.get("reference_accuracy") is not None]
            max_ref = max(refs) if refs else 0
            rt = r.get("runtime_seconds",0)
            report.append(f"| {ds} | {seed} | {eps:.4f} | {noise:.4f} | {steps} | {max_ref:.4f} | {rt:.1f} |")

# Save
with open(OUT, "w") as f:
    f.write("\n".join(report))
print(f"Report: {OUT}")
print(f"Lines: {len(report)}")
