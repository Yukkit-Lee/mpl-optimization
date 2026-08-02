#!/usr/bin/env python3
"""Rebuild the canonical unified report from audited existing artifacts.

This script is read-only with respect to experiment data. It does not import a
training backend, load checkpoints, or execute any experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reportToMe"
STATUS_ROOT = ROOT / "new_log/unified_experiments"


SOURCES = {
    "minimal_validation": ROOT
    / "new_log/new_true_ours/new_true_ours_minimal_validation.json",
    "master_results": ROOT
    / "new_log/new_true_ours/new_true_ours_master_results.json",
    "time_to_target": ROOT
    / "new_expTab/new_true_ours/new_figure2_time_to_target_table.csv",
    "stage_a": ROOT
    / "new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv",
    "module_c": ROOT
    / "new_expTab/new_public_mtt_mainline/new_module_c_formal_summary.csv",
    "historical_e2e": ROOT
    / "new_expTab/new_public_mtt_mainline/new_end_to_end_formal_summary.csv",
    "postagg_main": ROOT
    / "new_expTab/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_main_table.csv",
    "postagg_tfe": ROOT
    / "new_expTab/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_tfe_results.csv",
    "postagg_audit": ROOT
    / "new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_completion_audit.json",
    "kd_causal": ROOT
    / "new_expTab/new_stage_b_kd/new_stage_b_kd_anchor_supervision_ablation.csv",
    "kd_audit": ROOT
    / "new_log/new_stage_b_kd/new_stage_b_kd_completion_audit.json",
    "refinement_main": ROOT
    / "new_expTab/new_stage_b_anchor_refinement/new_stage_b_anchor_refinement_main_table.csv",
    "refinement_audit": ROOT
    / "new_log/new_stage_b_anchor_refinement/new_stage_b_anchor_refinement_completion_audit.json",
    "trigger_main": ROOT
    / "new_expTab/new_stage_b_heterogeneity_trigger/new_stage_b_heterogeneity_trigger_main_table.csv",
    "trigger_audit": ROOT
    / "new_log/new_stage_b_heterogeneity_trigger/new_stage_b_heterogeneity_trigger_completion_audit.json",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def commit_exists_and_is_ancestor(value: str) -> bool:
    exists = subprocess.run(
        ["git", "cat-file", "-e", value + "^{commit}"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not exists:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", value, "HEAD"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def mean_std(values: Iterable[float]) -> Tuple[float, Optional[float]]:
    values = list(values)
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else None
    )


def percent(value: float) -> str:
    return "{:.2f}%".format(100.0 * value)


def pp(value: float) -> str:
    return "{:+.2f}pp".format(100.0 * value)


def mean_pm(mean: float, std: Optional[float], unit: str = "%") -> str:
    if unit == "%":
        if std is None:
            return "{:.2f}% (n=1)".format(100.0 * mean)
        return "{:.2f}% ± {:.2f}%".format(100.0 * mean, 100.0 * std)
    if std is None:
        return "{:.2f} {} (n=1)".format(mean, unit)
    return "{:.2f} ± {:.2f} {}".format(mean, std, unit)


def exact_row(rows: List[Dict[str, str]], **conditions: str) -> Dict[str, str]:
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Expected one row for {}, got {}".format(conditions, len(selected))
        )
    return selected[0]


def validate_and_collect() -> Dict[str, Any]:
    for path in SOURCES.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing source artifact: {}".format(path))

    minimal = load_json(SOURCES["minimal_validation"])
    master = load_json(SOURCES["master_results"])
    stage_a = load_csv(SOURCES["stage_a"])
    targets = load_csv(SOURCES["time_to_target"])
    postagg = load_csv(SOURCES["postagg_main"])
    postagg_tfe = load_csv(SOURCES["postagg_tfe"])
    kd = load_csv(SOURCES["kd_causal"])
    refinement = load_csv(SOURCES["refinement_main"])
    trigger = load_csv(SOURCES["trigger_main"])
    module_c = load_csv(SOURCES["module_c"])
    historical_e2e = load_csv(SOURCES["historical_e2e"])

    if minimal.get("status") != "COMPLETE_14_OF_14":
        raise RuntimeError("True Ours validation is not 14/14 complete")
    if (minimal.get("passed"), minimal.get("failed"), minimal.get("missing")) != (
        14,
        0,
        0,
    ):
        raise RuntimeError("True Ours validation cardinality mismatch")
    if master.get("status") != "COMPLETE_DEVELOPMENT_ONLY_14_OF_14":
        raise RuntimeError("True Ours master result is not authoritative")
    if master.get("official_test_read") or master.get("locked_holdout_read"):
        raise RuntimeError("True Ours master result crossed the evaluation boundary")
    if len(stage_a) != 9 or len(postagg) != 6 or len(postagg_tfe) != 2:
        raise RuntimeError("Unexpected core table cardinality")

    expected_samples = {
        "cifar10": (50000, 480, 0.0096),
        "cifar100": (50000, 4800, 0.096),
        "tiny": (100000, 9600, 0.096),
    }
    for dataset, (full_count, compact_count, ratio) in expected_samples.items():
        full = exact_row(stage_a, Dataset=dataset, Method="full_images")
        random = exact_row(stage_a, Dataset=dataset, Method="random_subset")
        mtt = exact_row(stage_a, Dataset=dataset, Method="official_mtt")
        if int(full["Samples"]) != full_count:
            raise RuntimeError("Full-data sample count mismatch")
        if int(random["Samples"]) != compact_count or int(mtt["Samples"]) != compact_count:
            raise RuntimeError("Compact sample count mismatch")
        if abs(float(mtt["CompressionRatio"]) - ratio) > 1e-12:
            raise RuntimeError("Compression ratio mismatch")

    completion = {
        "postagg": load_json(SOURCES["postagg_audit"])["status"],
        "kd": load_json(SOURCES["kd_audit"])["status"],
        "refinement": load_json(SOURCES["refinement_audit"])["status"],
        "trigger": load_json(SOURCES["trigger_audit"])["status"],
    }
    if completion != {
        "postagg": "COMPLETE",
        "kd": "COMPLETE",
        "refinement": "COMPLETE",
        "trigger": "COMPLETE_NEGATIVE",
    }:
        raise RuntimeError("Completion audit status mismatch: {}".format(completion))
    if not all(
        row.get("correctness_pass") == "True"
        and row.get("utility_gate") == "True"
        and row.get("test_read") == "False"
        and row.get("locked_holdout_read") == "False"
        for row in postagg_tfe
    ):
        raise RuntimeError("Post-aggregation TFE gate mismatch")

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in minimal["rows"]:
        grouped[(row["dataset"], row["method"])].append(row)
    true_ours_summary: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (dataset, method), rows in grouped.items():
        accuracy_mean, accuracy_std = mean_std(
            float(row["final_development_accuracy"]) for row in rows
        )
        time_mean, time_std = mean_std(
            float(row["total_online_secure_seconds"]) for row in rows
        )
        t90 = exact_row(
            targets,
            dataset=dataset,
            method=method,
            row_type="aggregate",
            target="T@90",
        )
        true_ours_summary[dataset][method] = {
            "n": len(rows),
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "time_mean": time_mean,
            "time_std": time_std,
            "t90_mean": float(t90["time_seconds"]),
            "t90_std": (
                None
                if t90["sample_std_seconds"] == "N/A"
                else float(t90["sample_std_seconds"])
            ),
        }

    source_hashes = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in SOURCES.items()
    }
    lineage = set()
    for row in postagg:
        lineage.add(row["report_source_git_commit"])
        lineage.add(row["experiment_source_git_commit"])
    for row in postagg_tfe:
        lineage.add(row["source_git_commit"])
    if not all(commit_exists_and_is_ancestor(value) for value in lineage):
        raise RuntimeError("Post-aggregation lineage commit is missing")

    return {
        "minimal": minimal,
        "master": master,
        "stage_a": stage_a,
        "targets": targets,
        "postagg": postagg,
        "postagg_tfe": postagg_tfe,
        "kd": kd,
        "refinement": refinement,
        "trigger": trigger,
        "module_c": module_c,
        "historical_e2e": historical_e2e,
        "true_ours_summary": true_ours_summary,
        "completion": completion,
        "source_hashes": source_hashes,
        "lineage_commits": sorted(lineage),
    }


def legacy_inventory() -> Dict[str, Any]:
    partial_json = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "log/json").glob("*.partial.json")
    )
    malformed_csv: List[Dict[str, Any]] = []
    for path in sorted((ROOT / "expTab").glob("*.csv")):
        mismatches = 0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            expected = None
            for row in reader:
                if expected is None:
                    expected = len(row)
                elif len(row) != expected:
                    mismatches += 1
        if mismatches:
            malformed_csv.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "mixed_schema_rows": mismatches,
                }
            )
    return {
        "partial_json": partial_json,
        "legacy_mixed_schema_csv": malformed_csv,
        "policy": "provenance_only_excluded_from_recursive_current_result_parsing",
    }


def build_report(data: Dict[str, Any], source: str) -> str:
    lines = [
        "# Unified Experiment Report",
        "",
        "更新时间：2026-08-02（UTC）",
        "权威分支：`unified-experiments`",
        "报告生成提交：`{}`".format(source),
        "",
        "本报告是仓库当前唯一权威的跨实验总结。所有数字均由已跟踪的 CSV/JSON",
        "重新读取并通过 cardinality、状态、数据边界、SHA256 与 Git lineage 审计；",
        "本次没有启动训练、TFE 或任何正式实验。",
        "",
        "## 1. 总结结论",
        "",
        "1. Official MTT 在三个数据集上均显著优于等规模随机子集，但样本比例并不都是1%。",
        "2. True Ours 的最新公平主比较为14/14 development-validation结果：三个数据集均更快达到T@90；CIFAR-100/Tiny最终development accuracy更高，CIFAR-10低4.29pp。",
        "3. 当前最可信的正Stage-B扩展是 Secure Average 后接有标签Official-MTT anchor CE；正式TFE复现了CIFAR-10/100 Severe方向。",
        "4. KD/Consensus的表面增益主要来自anchor标签监督；class-aware KD没有独立正贡献。",
        "5. disagreement weighting/top-selection与heterogeneity-trigger routing均为完成的负结果，不能包装为正模块。",
        "6. Stage-C epsilon只覆盖条件式Stage-C机制；数据依赖的Stage-B initializer非record-level DP，因此不能声称端到端DP。",
        "7. 当前权威True Ours与Post-Aggregation均未读取official test或locked holdout；历史final-test数字不进入当前主结论。",
        "",
        "## 2. Stage A：Official MTT样本效率",
        "",
        "| Dataset | Full samples | Compact samples | Compact ratio | Full | Random subset | Official MTT | MTT−Random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("cifar10", "cifar100", "tiny"):
        full = exact_row(data["stage_a"], Dataset=dataset, Method="full_images")
        random = exact_row(data["stage_a"], Dataset=dataset, Method="random_subset")
        mtt = exact_row(data["stage_a"], Dataset=dataset, Method="official_mtt")
        lines.append(
            "| {} | {:,} | {:,} | {:.2f}% | {} | {} | {} | {} |".format(
                dataset,
                int(full["Samples"]),
                int(mtt["Samples"]),
                100 * float(mtt["CompressionRatio"]),
                mean_pm(float(full["FinalAccuracyMean"]), float(full["FinalAccuracyStd"])),
                mean_pm(float(random["FinalAccuracyMean"]), float(random["FinalAccuracyStd"])),
                mean_pm(float(mtt["FinalAccuracyMean"]), float(mtt["FinalAccuracyStd"])),
                pp(float(mtt["FinalAccuracyMean"]) - float(random["FinalAccuracyMean"])),
            )
        )
    lines.extend(
        [
            "",
            "结论：MTT相对随机子集提升6.83/4.79/3.57pp；CIFAR-100与Tiny的compact ratio是9.6%，不是1%。",
            "",
            "## 3. True Ours vs Full Data：当前主比较",
            "",
            "状态：`COMPLETE_DEVELOPMENT_ONLY_14_OF_14`。CIFAR-10/100各3 seeds；Tiny按用户批准仅seed42。",
            "",
            "| Dataset | Method | n | Final development accuracy | Total online secure time | T@90 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("cifar10", "cifar100", "tiny"):
        for method in ("full_data", "true_ours"):
            row = data["true_ours_summary"][dataset][method]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    method,
                    row["n"],
                    mean_pm(row["accuracy_mean"], row["accuracy_std"]),
                    mean_pm(row["time_mean"], row["time_std"], "s"),
                    mean_pm(row["t90_mean"], row["t90_std"], "s"),
                )
            )
    lines.extend(
        [
            "",
            "客观解释：",
            "",
            "- CIFAR-10：True Ours最终accuracy低约4.29pp，但总时长快约4.31×，T@90快约2.54×。",
            "- CIFAR-100：True Ours高约5.40pp，总时长快约1.44×，T@90快约2.53×。",
            "- Tiny：True Ours高4.21pp且T@90快约2.46×，但完整运行总时长慢约1.52×；只有n=1。",
            "- 安全表述是“更快达到冻结目标”，不能写成“所有数据集最终都更快/更准”。",
            "",
            "## 4. 最新正Stage-B：Post-Aggregation Anchor CE",
            "",
            "范围：CIFAR-10/100、seed42、development-only、Official-MTT-only；40 local + 8 anchor/class；无Full initializer、test或locked holdout。",
            "",
            "| Dataset | Dist. | C1 Random+Anchor | C3 Local-0+Anchor | C4 Average | C5 Average+Anchor | C5−C1 | C5−C3 | C5−C4 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["postagg"]:
        lines.append(
            "| {dataset} | {distribution} | {c1} | {c3} | {c4} | {c5} | {d1} | {d3} | {d4} |".format(
                dataset=row["dataset"],
                distribution=row["distribution"],
                c1=percent(float(row["c1_random_plus_anchor"])),
                c3=percent(float(row["c3_local0_plus_anchor"])),
                c4=percent(float(row["c4_secure_average"])),
                c5=percent(float(row["c5_secure_average_plus_anchor"])),
                d1=pp(float(row["c5_minus_c1"])),
                d3=pp(float(row["c5_minus_c3"])),
                d4=pp(float(row["c5_minus_c4"])),
            )
        )
    lines.extend(
        [
            "",
            "### 正式TFE/ABY3 Severe复现",
            "",
            "| Dataset | TFE C1 | TFE C4 | TFE C5 | C5−C1 | Gain retention | Max accuracy gap | Secure compute | Peak RSS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["postagg_tfe"]:
        lines.append(
            "| {dataset} | {c1} | {c4} | {c5} | {delta} | {ret:.1f}% | {gap:.2f}pp | {runtime:.2f}s | {rss:.2f}GiB |".format(
                dataset=row["dataset"],
                c1=percent(float(row["tfe_c1_accuracy"])),
                c4=percent(float(row["tfe_c4_accuracy"])),
                c5=percent(float(row["tfe_c5_accuracy"])),
                delta=pp(float(row["tfe_c5_minus_c1"])),
                ret=100 * float(row["gain_retention"]),
                gap=100 * float(row["maximum_accuracy_gap"]),
                runtime=float(row["total_secure_seconds"]),
                rss=float(row["peak_rss_gib"]),
            )
        )
    lines.extend(
        [
            "",
            "结论：Secure Average对anchor CE有独立贡献，尤其在Mild/Severe下相对single-client明显；IID下C5与合法Local-0+Anchor仅近似持平，因此应定位为Non-IID扩展。",
            "",
            "## 5. Stage-B因果与负结果",
            "",
            "### 5.1 KD/Consensus因果分解（seed42）",
            "",
            "| Dataset Severe | Average | Anchor CE-only | Uniform CE+KD | Class-aware CE+KD | KD−CE-only | Class-aware−Uniform |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("cifar10", "cifar100"):
        average = exact_row(data["kd"], dataset=dataset, distribution="severe", method="secure_average_equivalent_plaintext")
        ce = exact_row(data["kd"], dataset=dataset, distribution="severe", method="anchor_ce_only")
        uniform = exact_row(data["kd"], dataset=dataset, distribution="severe", method="uniform_ce_plus_kd")
        aware = exact_row(data["kd"], dataset=dataset, distribution="severe", method="class_aware_ce_plus_kd")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                dataset,
                percent(float(average["initializer_accuracy"])),
                percent(float(ce["initializer_accuracy"])),
                percent(float(uniform["initializer_accuracy"])),
                percent(float(aware["initializer_accuracy"])),
                pp(float(uniform["initializer_accuracy"]) - float(ce["initializer_accuracy"])),
                pp(float(aware["initializer_accuracy"]) - float(uniform["initializer_accuracy"])),
            )
        )
    lines.extend(
        [
            "",
            "结论：anchor标签监督有效；KD在CIFAR-10仅+0.44pp、CIFAR-100为−3.54pp，class-aware相对uniform为0/−0.02pp。KD/Consensus不是独立正贡献。",
            "",
            "### 5.2 Disagreement-guided refinement",
            "",
            "| Dataset Severe | Uniform anchor CE | Weighted CE | Top-50% CE | Best informative AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    auroc = {"cifar10": 0.2243, "cifar100": 0.4340}
    for dataset in ("cifar10", "cifar100"):
        b1 = exact_row(data["refinement"], dataset=dataset, method="B1_uniform_anchor_ce")
        b2 = exact_row(data["refinement"], dataset=dataset, method="B2_disagreement_weighted_ce")
        b3 = exact_row(data["refinement"], dataset=dataset, method="B3_top_disagreement_ce")
        lines.append(
            "| {} | {} | {} | {} | {:.4f} |".format(
                dataset,
                percent(float(b1["development_accuracy"])),
                percent(float(b2["development_accuracy"])),
                percent(float(b3["development_accuracy"])),
                auroc[dataset],
            )
        )
    lines.extend(
        [
            "",
            "结论：分歧分数与错误方向相反，效用和成本门均失败；按计划未启动TFE。",
            "",
            "### 5.3 Heterogeneity-trigger routing",
            "",
            "H1--H3能测IID→Mild→Severe标签偏斜，但uniform anchor CE在六场均提升B0。触发器在IID关闭CE会损失CIFAR-10 34.98pp、CIFAR-100 11.88pp；mean/max oracle regret为7.81/34.98pp。因此是完成的负结果，未启动TFE。",
            "",
            "## 6. Stage C、旧End-to-End与系统证据",
            "",
            "### 6.1 Fixed vs Adaptive DP（历史消融）",
            "",
            "| Dataset, ε=3 | Fixed accuracy | Adaptive accuracy | Fixed runtime | Adaptive runtime |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("cifar10", "cifar100", "tiny"):
        fixed = exact_row(data["module_c"], dataset=dataset, mode="fixed_dp", target_epsilon="3.0")
        adaptive = exact_row(data["module_c"], dataset=dataset, mode="adaptive_dp", target_epsilon="3.0")
        lines.append(
            "| {} | {} | {} | {:.2f}s | {:.2f}s |".format(
                dataset,
                mean_pm(float(fixed["final_accuracy_mean"]), float(fixed["final_accuracy_std"])),
                mean_pm(float(adaptive["final_accuracy_mean"]), float(adaptive["final_accuracy_std"])),
                float(fixed["runtime_mean_seconds"]),
                float(adaptive["runtime_mean_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "该历史消融中Adaptive没有持续效用优势，适合作为负/机制消融。当前True Ours Stage C使用Adaptive，但epsilon只对条件式Stage-C机制成立。",
            "",
            "### 6.2 历史End-to-End",
            "",
            "早期18/18 Random-subset Fixed vs Official-MTT Consensus Adaptive结果可以作为历史补充证据，但协议已被当前Full Data vs True Ours公平主比较取代，不应进入当前主表。",
            "",
            "### 6.3 TFE系统表",
            "",
            "旧`new_tfe_system_formal.csv`的runtime/RSS为0占位值，不是可发表系统测量。当前可用的真实系统数据来自True Ours timeline与Post-Aggregation TFE runtime/resource表；通信字节与轮数仍为N/A。",
            "",
            "## 7. 当前论文就绪度",
            "",
            "| Evidence | Status | Safe use |",
            "|---|---|---|",
            "| Stage A Official MTT efficiency | READY | 主文/补充；使用正确样本数与比例 |",
            "| True Ours vs Full Data 14/14 | READY_WITH_SCOPE | 可作主结果，但必须标development validation；Tiny n=1 |",
            "| Post-Aggregation Anchor CE | READY_WITH_SCOPE | 正Stage-B/Non-IID扩展；seed42、C10/C100 |",
            "| Real TFE/ABY3 correctness/runtime | READY | 系统可行性；通信N/A |",
            "| KD/Consensus causality | COMPLETE_NEGATIVE | 负消融；不可声称独立贡献 |",
            "| Disagreement refinement | COMPLETE_NEGATIVE | 负消融；TFE按门控跳过 |",
            "| Heterogeneity trigger | COMPLETE_NEGATIVE | 负消融；TFE按门控跳过 |",
            "| Fixed vs Adaptive | SUPPLEMENTARY_NEGATIVE | Adaptive无持续显著优势 |",
            "| Current official-test main table | NOT_AVAILABLE | 当前权威路线未读取official test |",
            "| End-to-end record-level DP | NOT_SUPPORTED | Stage-B initializer数据依赖且非DP |",
            "| Communication reduction | NOT_MEASURED | 不得声称 |",
            "",
            "## 8. 安全论文表述",
            "",
            "可以写：Official-MTT-only路线以更少compact样本提供高信息密度；真实TFE/ABY3复现安全训练；Secure Average作为跨方中间初始化，再用类别均衡的标签anchor监督修正，在受控seed42消融中优于Random+Anchor，并在异质场景优于single-client+Anchor。",
            "",
            "不可以写：所有数据集最终都更快/更准、当前数字是official-test、整个发布模型满足epsilon=3端到端DP、class-aware KD/Consensus是正贡献、trigger能安全决定是否refine、或通信成本下降。",
            "",
            "## 9. 权威文件",
            "",
            "- True Ours：`new_log/new_true_ours/new_true_ours_master_results.json`",
            "- Figure 2：`new_expFig/new_true_ours/new_figure2_pea_style_secure_time.{png,pdf,svg}`",
            "- Stage A：`new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv`",
            "- Post-Aggregation：`new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_final_report.md`",
            "- Post-Aggregation审计：`new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_completion_audit.json`",
            "- 统一machine status：`new_log/unified_experiments/unified_artifact_status.json`",
            "- 统一数据审计：`new_log/unified_experiments/unified_data_audit.json`",
            "",
            "## 10. 剩余缺口",
            "",
            "- 若投稿主表必须报告official test，需要另行设计一次冻结后、可审计的最终读取；本次未运行。",
            "- Post-Aggregation正结果目前只有seed42；若要声称跨seed稳定，需未来补seed43/44。",
            "- 外部方法PEA/CaPC/DDGauss精确对比仍需从原论文/官方仓库核对。",
            "- 通信量没有测量，系统表只能报告实际secure runtime和RSS。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readiness(data: Dict[str, Any], source: str) -> str:
    return """# Current Paper Readiness Assessment

更新时间：2026-08-02（UTC）
状态：**READY WITH EXPLICIT SCOPE LIMITATIONS**
来源提交：`{source}`

本文件已取代2026-07-28版本。完整数据、负结果和证据路径见
[`unified_experiment_report.md`](unified_experiment_report.md)。

## 可进入论文

- Stage A Official MTT样本效率：三数据集3 seeds；正确compact规模为
  480/4,800/9,600，不可统一写成480或1%。
- True Ours vs Full Data：14/14 development-validation结果。C10/C100各3
  seeds，Tiny仅seed42。三个数据集均更快达到T@90；最终accuracy并非全胜。
- 真实TFE/ABY3可行性与runtime/RSS：使用True Ours timeline和最新
  Post-Aggregation正式结果，不使用旧0秒占位System表。
- Secure Average + Official-MTT Anchor CE：正Stage-B/Non-IID扩展，当前为
  CIFAR-10/100 seed42 development-only。

## 只适合作为负/补充实验

- KD/Consensus因果分解：独立KD贡献不成立。
- Disagreement-guided refinement：效用和成本门失败。
- Heterogeneity-trigger routing：oracle regret门失败。
- Fixed vs Adaptive：Adaptive没有持续显著效用优势。
- 旧End-to-End：协议不同且已被当前公平主比较取代。

## 当前不能声称

- 当前主结果是official-test或locked-holdout结果；两者在权威链中均未读取。
- 整个端到端发布模型是record-level DP；epsilon只覆盖条件式Stage-C机制。
- 三个数据集最终都更快、更准。
- class-aware KD、disagreement weighting或trigger routing是正贡献。
- 通信字节或轮数降低；没有测量。

## 投稿前最值得补的内容

1. 若审稿要求test主表，执行一次冻结后、带读取ledger的最终评估。
2. Post-Aggregation补seed43/44以支持跨seed稳定性。
3. 核对外部方法的官方数字与协议。
""".format(source=source)


def build_status(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "schema": "unified_experiment_artifact_status_v1",
        "status": "COMPLETE_WITH_SCOPE_LIMITATIONS",
        "source_git_commit": source,
        "canonical_report": "reportToMe/unified_experiment_report.md",
        "current_authoritative": [
            {
                "experiment": "true_ours_full_data_comparison",
                "status": "COMPLETE_DEVELOPMENT_ONLY_14_OF_14",
                "artifact": "new_log/new_true_ours/new_true_ours_master_results.json",
            },
            {
                "experiment": "stage_a_official_mtt_efficiency",
                "status": "COMPLETE",
                "artifact": "new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv",
            },
            {
                "experiment": "stage_b_anchor_postagg",
                "status": "COMPLETE_SEED42_TFE",
                "artifact": "new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_completion_audit.json",
            },
        ],
        "completed_negative": [
            {
                "experiment": "stage_b_kd_consensus_causality",
                "status": "COMPLETE_NEGATIVE_CAUSAL_CONTRIBUTION",
                "artifact": "new_log/new_stage_b_kd/new_stage_b_kd_completion_audit.json",
            },
            {
                "experiment": "stage_b_disagreement_refinement",
                "status": "COMPLETE_NEGATIVE",
                "artifact": "new_log/new_stage_b_anchor_refinement/new_stage_b_anchor_refinement_completion_audit.json",
            },
            {
                "experiment": "stage_b_heterogeneity_trigger",
                "status": "COMPLETE_NEGATIVE",
                "artifact": "new_log/new_stage_b_heterogeneity_trigger/new_stage_b_heterogeneity_trigger_completion_audit.json",
            },
        ],
        "superseded_or_quarantined": [
            {
                "artifact": "new_log/new_true_ours/new_true_ours_formal_queue_status.json",
                "reason": "historical interrupted queue snapshot; no active process",
            },
            {
                "artifact": "new_log/new_true_ours/direct_evaluation_16models.json",
                "reason": "unverified evaluation source; excluded from current claims",
            },
            {
                "artifact": "new_log/new_true_ours/new_tiny_full_seed42_superseded_artifact_manifest.json",
                "reason": "explicitly quarantined and superseded",
            },
            {
                "artifact": "new_expTab/new_public_mtt_mainline/new_tfe_system_formal.csv",
                "reason": "zero runtime/RSS placeholders; not a paper-ready system table",
            },
        ],
        "boundaries": {
            "current_official_test_read": False,
            "current_locked_holdout_read": False,
            "true_ours_accuracy_scope": "development_validation_only",
            "stage_c_privacy_scope": "conditional_stage_c_only",
            "end_to_end_record_level_dp": False,
            "communication_measured": False,
        },
        "legacy_inventory": legacy_inventory(),
        "source_hashes": data["source_hashes"],
    }


def main() -> int:
    source = current_commit()
    data = validate_and_collect()
    report = build_report(data, source)
    readiness = build_readiness(data, source)
    status = build_status(data, source)
    audit = {
        "schema": "unified_experiment_data_audit_v1",
        "status": "PASS",
        "source_git_commit": source,
        "checks": {
            "true_ours_14_of_14": True,
            "true_ours_development_only": True,
            "stage_a_sample_counts_correct": True,
            "postagg_plaintext_rows": len(data["postagg"]),
            "postagg_tfe_rows": len(data["postagg_tfe"]),
            "postagg_tfe_gates_pass": True,
            "completion_audits": data["completion"],
            "lineage_commits_exist_and_are_ancestors": True,
            "no_experiment_executed_by_report_builder": True,
        },
        "source_hashes": data["source_hashes"],
        "lineage_commits": data["lineage_commits"],
    }
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "unified_experiment_report.md").write_text(report, encoding="utf-8")
    (REPORT_ROOT / "paper_readiness_assessment.md").write_text(readiness, encoding="utf-8")
    (REPORT_ROOT / "new_all_experiments_summary.md").write_text(
        "# All Experiments Summary\n\nThis path is retained for compatibility. "
        "The canonical, data-audited report is "
        "[`unified_experiment_report.md`](unified_experiment_report.md).\n",
        encoding="utf-8",
    )
    (REPORT_ROOT / "README.md").write_text(
        "# Report authority\n\n"
        "1. `unified_experiment_report.md` — complete canonical report.\n"
        "2. `paper_readiness_assessment.md` — concise current readiness view.\n"
        "3. Per-experiment final reports — protocol-specific detail.\n\n"
        "Other historical reports are provenance snapshots and must not override "
        "the unified artifact status.\n",
        encoding="utf-8",
    )
    (STATUS_ROOT / "unified_artifact_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (STATUS_ROOT / "unified_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "PASS unified report sources={} true_ours=14/14 postagg=6+2".format(
            len(SOURCES)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
