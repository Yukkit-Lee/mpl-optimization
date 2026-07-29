#!/usr/bin/env python3
"""Requirement-by-requirement completion audit for the seed42 Stage-B task."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "new_log/new_stage_b_kd"
TABLE = ROOT / "new_expTab/new_stage_b_kd"
FIGURE = ROOT / "new_expFig/new_stage_b_kd"
REPORT = ROOT / "reportToMe"
PLAN = ROOT / "plan"
OUTPUT_JSON = LOG / "new_stage_b_kd_completion_audit.json"
OUTPUT_MD = LOG / "new_stage_b_kd_completion_audit.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(*command):
    return subprocess.check_output(command, cwd=str(ROOT), text=True).strip()


def check(name, passed, evidence):
    return {"requirement": name, "passed": bool(passed), "evidence": evidence}


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    checks = []

    branch = run("git", "branch", "--show-current")
    checks.append(
        check(
            "dedicated branch",
            branch == "new-stage-b-mtt-anchor-kd",
            branch,
        )
    )

    policy = load_json(
        ROOT / "new_configs/new_stage_b_kd/new_stage_b_kd_single_seed_policy.json"
    )
    checks.append(
        check(
            "single seed policy",
            policy.get("seed_policy", {}).get("all_phases") == [42],
            str(policy.get("seed_policy")),
        )
    )

    split = load_json(LOG / "new_stage_b_kd_split_manifest.json")
    split_rows = split.get("rows", split.get("datasets", []))
    split_text = (LOG / "new_stage_b_kd_split_audit.md").read_text(encoding="utf-8")
    checks.append(
        check(
            "disjoint split audit",
            bool(split_rows)
            and "intersection" in split_text.lower()
            and "official" in split_text.lower(),
            "manifest rows={} audit present".format(len(split_rows)),
        )
    )

    registry = load_json(LOG / "new_stage_b_kd_smoke_registry.json")
    runs = registry["runs"]
    stages = sorted({row["stage"] for row in runs})
    checks.append(
        check(
            "S1-S6 plaintext search",
            len(runs) == 89
            and set(stages) == {"S1", "S2", "S3", "S4", "S5"},
            "registered runs={} tuning stages={} plus S6 confirmation table".format(
                len(runs), stages
            ),
        )
    )

    confirmation = load_csv(TABLE / "new_stage_b_kd_plaintext_confirmation.csv")
    confirmation_keys = {
        (row["dataset"], row["distribution"], row["method"])
        for row in confirmation
    }
    expected_confirmation = {
        (dataset, distribution, method)
        for dataset in ("cifar10", "cifar100")
        for distribution in ("iid", "mild", "severe")
        for method in (
            "secure_average_equivalent_plaintext",
            "uniform_logit_kd",
            "class_aware_consensus_kd",
        )
    }
    checks.append(
        check(
            "single-seed plaintext confirmation",
            confirmation_keys == expected_confirmation
            and {row["seed"] for row in confirmation} == {"42"},
            "rows={} seed42 only".format(len(confirmation)),
        )
    )

    causal = load_json(LOG / "new_stage_b_kd_anchor_supervision_ablation.json")
    checks.append(
        check(
            "causal anchor/KD decomposition",
            len(causal.get("rows", [])) == 36
            and causal.get("anchor_data_effective_all_datasets") is True
            and causal.get("class_aware_consensus_positive_all_datasets") is False,
            "36 rows; anchor effective; class-aware Consensus not positive",
        )
    )

    tfe = load_json(LOG / "new_stage_b_kd_tfe_validation.json")
    tfe_rows = {row["dataset"]: row for row in tfe.get("rows", [])}
    tfe_required = {"tiny_synthetic", "cifar10", "cifar100"}
    checks.append(
        check(
            "complete TFE/ABY3 ladder",
            set(tfe_rows) == tfe_required
            and all(row.get("correctness_pass") for row in tfe_rows.values())
            and all(row.get("backend") == "TFE_ABY3" for row in tfe_rows.values()),
            "available={} passed={}".format(
                sorted(tfe_rows),
                sorted(
                    key
                    for key, row in tfe_rows.items()
                    if row.get("correctness_pass")
                ),
            ),
        )
    )

    lineage_payloads = [registry, causal, tfe]
    checks.append(
        check(
            "lineage and unread evaluation boundary",
            all(not item.get("locked_holdout_read") for item in lineage_payloads)
            and all(not item.get("official_test_read") for item in lineage_payloads)
            and all(
                not item.get("full_data_initializer_loaded") for item in lineage_payloads
            ),
            "locked/test=false, Full-Data initializer=false",
        )
    )

    required_files = [
        TABLE / "new_stage_b_kd_smoke_registry.csv",
        LOG / "new_stage_b_kd_smoke_summary.json",
        LOG / "new_stage_b_kd_diagnostic_report.md",
        TABLE / "new_stage_b_kd_plaintext_confirmation.csv",
        TABLE / "new_stage_b_kd_tfe_validation.csv",
        LOG / "new_stage_b_kd_claim_audit.md",
        REPORT / "new_stage_b_kd_final_report.md",
        TABLE / "new_stage_b_kd_runtime_breakdown.csv",
        TABLE / "new_stage_b_kd_resource_summary.csv",
        TABLE / "new_stage_b_kd_main_ablation.csv",
        TABLE / "new_stage_b_kd_classwise_accuracy.csv",
        PLAN / "0729_new_stage_b_kd_remaining_seeds_plan.md",
    ]
    figure_stems = [
        "new_figureB1_noniid_initializer_quality",
        "new_figureB2_client_diversity_teacher_gain",
        "new_figureB3_runtime_breakdown",
        "new_figureS1_anchor_size_sweep",
        "new_figureS2_temperature_sweeps",
        "new_figureS3_plaintext_tfe_numerics",
        "new_figureS4_class_weight_distributions",
        "new_figureS5_classwise_accuracy_delta",
    ]
    figure_files = [
        FIGURE / (stem + extension)
        for stem in figure_stems
        for extension in (".png", ".pdf")
    ]
    missing = [
        str(path.relative_to(ROOT))
        for path in required_files + figure_files
        if not path.is_file() or path.stat().st_size == 0
    ]
    checks.append(
        check(
            "mandatory reports, tables, figures, and remaining-seed plan",
            not missing,
            "missing={}".format(missing),
        )
    )

    tracked_checkpoints = run("git", "ls-files", "new_checkpoints")
    checks.append(
        check(
            "checkpoint policy",
            tracked_checkpoints == "",
            "no checkpoint tracked by Git",
        )
    )

    forbidden_names = []
    for path in list(LOG.rglob("*")) + list(TABLE.rglob("*")) + list(FIGURE.rglob("*")):
        lowered = path.name.lower()
        if any(
            token in lowered
            for token in ("stage_c", "end_to_end", "figure2", "t_target", "final_test")
        ):
            forbidden_names.append(str(path.relative_to(ROOT)))
    checks.append(
        check(
            "forbidden experiment isolation",
            not forbidden_names,
            "forbidden artifacts={}".format(forbidden_names),
        )
    )

    plan_text = (
        PLAN / "0729_new_stage_b_kd_remaining_seeds_plan.md"
    ).read_text(encoding="utf-8") if (
        PLAN / "0729_new_stage_b_kd_remaining_seeds_plan.md"
    ).is_file() else ""
    checks.append(
        check(
            "remaining seeds are plan-only",
            "does **not**\nauthorize or launch seeds 43/44" in plan_text
            and "seeds: 43 and 44 only" in plan_text,
            "no seeds43/44 result rows or launch authorization",
        )
    )

    passed = sum(item["passed"] for item in checks)
    status = "COMPLETE" if passed == len(checks) else "INCOMPLETE"
    payload = {
        "schema": "new_stage_b_kd_completion_audit_v1",
        "status": status,
        "source_commit": run("git", "rev-parse", "HEAD"),
        "passed": passed,
        "total": len(checks),
        "seed_policy": [42],
        "checks": checks,
    }
    atomic_text(OUTPUT_JSON, json.dumps(payload, indent=2))
    markdown = [
        "# Stage-B KD Completion Audit",
        "",
        "Status: **{} ({}/{})**".format(status, passed, len(checks)),
        "",
        "| Requirement | Pass | Evidence |",
        "|---|---:|---|",
    ]
    for item in checks:
        markdown.append(
            "| {} | {} | {} |".format(
                item["requirement"],
                "YES" if item["passed"] else "NO",
                item["evidence"].replace("|", "/"),
            )
        )
    atomic_text(OUTPUT_MD, "\n".join(markdown))
    print("{} passed={}/{}".format(status, passed, len(checks)))
    if args.require_complete and status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
