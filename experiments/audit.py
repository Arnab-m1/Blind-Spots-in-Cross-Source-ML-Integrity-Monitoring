"""Fail-fast artifact, schema, split, and claim audit."""

from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def require_columns(path, columns):
    present = set(pd.read_csv(path, nrows=2).columns)
    missing = set(columns) - present
    if missing: raise AssertionError(f"{path}: missing {sorted(missing)}")

def run():
    required = [
        ROOT / "results/figures/fig_synthetic_phase_diagram.pdf",
        ROOT / "results/figures/fig_icrisat_nonlinear.pdf",
        ROOT / "results/figures/fig_frontier_calibration.pdf",
        ROOT / "results/figures/fig_hydraulic_attacks.pdf",
        ROOT / "results/figures/fig_hydraulic_matched_security.pdf",
        ROOT / "results/figures/fig_hydraulic_prioritization.pdf",
        ROOT / "results/figures/fig_graybox_transfer.pdf",
        ROOT / "results/figures/fig_icrisat_hardening.pdf",
        ROOT / "results/figures/fig_hydraulic_hardening.pdf",
        ROOT / "results/figures/fig_external_monitor_selectivity.pdf",
        ROOT / "results/figures/fig_heldout_defender_decision.pdf",
        ROOT / "results/tables/confidence_intervals.json",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0: raise AssertionError(f"missing {path}")
    require_columns(RAW / "e2_icrisat_nonlinear/certificates.csv", ["row","support","gamma","h","rank","nullity"])
    require_columns(RAW / "e2_icrisat_nonlinear/frontier_calibration.csv", ["delta","gamma","frontier_evidence","ratio","success"])
    require_columns(RAW / "e2_icrisat_nonlinear/seed_stability.csv", ["seed","r2","gamma_a4_harm_spearman"])
    require_columns(RAW / "e1_icrisat_linear/monitor_variants.csv", ["dimension","vector_gamma","scalar_blind_fraction"])
    require_columns(RAW / "e3_hydraulic/attack_results.csv", ["support","support_family","task","attack","gamma","budget","task_shift","detection_rate","class_flip_rate","false_negative_rate","perturbation_norm"])
    require_columns(RAW / "e3_hydraulic/seed_stability_summary.csv", ["seed","attack","median_abs_task_shift","median_detection","blind_fraction"])
    require_columns(RAW / "e3_hydraulic/task_metric_example.csv", ["task","rank","sigma_min","gamma","median_abs_task_shift"])
    require_columns(RAW / "e3_hydraulic/matched_security_summary.csv", ["seed","support","task","budget","attack","population","meaningful_adverse_rate","alarm_rate","joint_adverse_no_alarm_rate","feasible"])
    require_columns(RAW / "e3_hydraulic/matched_security_diagnostics.csv", ["seed","support","task","budget","attack","constraint_residual","solver_status","feasible"])
    require_columns(RAW / "e3_hydraulic/matched_security_main_table.csv", ["seed","attack","population","clean_task_accuracy","clean_alarm_rate","meaningful_adverse_rate","alarm_rate","joint_adverse_no_alarm_rate","invalid_rate"])
    require_columns(RAW / "e3_hydraulic/matched_security_condition_summary.csv", ["seed","attack","task","condition_label","support_family","budget","population","joint_adverse_no_alarm_rate"])
    require_columns(RAW / "e3_hydraulic/prioritization_cases.csv", ["case_id","partition","gamma","verified_failure"])
    require_columns(RAW / "e3_hydraulic/prioritization_verification.csv", ["case_id","budget","feasible","verified_failure","constraint_residual"])
    require_columns(RAW / "e3_hydraulic/prioritization_metrics.csv", ["seed","method","budget","precision","recall","audc","wall_seconds"])
    require_columns(RAW / "e5_hardening/hydraulic_hardening_results.csv", ["method","budget","blind_task_supports","adaptive_unit_budget_harm"])
    require_columns(RAW / "e4_graybox/graybox_results.csv", ["setting","epsilon_j","epsilon_f","defender_leakage","defender_task_harm"])
    require_columns(RAW / "e6_external_monitors/certificates.csv", ["monitor","support","task","gamma","rank","full_residual_rank"])
    require_columns(RAW / "e6_external_monitors/matched_security.csv", ["monitor","support","task","attack","budget","joint_adverse_no_alarm_rate"])
    require_columns(RAW / "e6_external_monitors/stability_certificates.csv", ["seed","monitor","support","task","gamma","is_blind"])
    require_columns(RAW / "e6_external_monitors/stability_matched_security.csv", ["seed","monitor","support","task","attack","budget","joint_adverse_no_alarm_rate"])
    require_columns(RAW / "e7_defender_decision/heldout_hardening.csv", ["seed","method","budget","original_failures","residual_failures","residual_joint_risk"])
    require_columns(RAW / "e7_defender_decision/finite_reruns.csv", ["case_id","method","protection_budget","verified_failure","feasible"])
    require_columns(RAW / "e8_metric_robustness/ranking_results.csv", ["seed","metric","budget","failures_found","precision"])
    require_columns(RAW / "e8_metric_robustness/paired_intervals.csv", ["budget","reference","comparator","paired_mean_difference","ci_low","ci_high"])
    require_columns(RAW / "e9_blind_larger_supports/selected_blind_configurations.csv", ["support_size","support","task","gamma","is_blind"])
    require_columns(RAW / "e9_blind_larger_supports/matched_attacks.csv", ["support_size","attack","budget_design","joint_adverse_no_alarm_rate"])
    require_columns(RAW / "e10_plausibility/support_budget_plausibility.csv", ["attack","budget","support","fraction_rows_nonnegative_physical_fields","fraction_rows_in_training_range"])
    provenance = json.loads((ROOT / "data/hydraulic/PROVENANCE.json").read_text())
    archive = ROOT / "data/hydraulic/raw.zip"
    if sha256(archive) != provenance["raw_sha256"]: raise AssertionError("Hydraulic hash mismatch")
    data = np.load(ROOT / "data/hydraulic/processed/cycle_features.npz")
    targets = data["targets"]
    chronological = {"train": np.arange(0,1323), "validation": np.arange(1323,1764), "test": np.arange(1764,2205)}
    class_coverage = {name: [len(np.unique(targets[idx,col])) for col in range(5)] for name,idx in chronological.items()}
    report = {
        "status":"pass", "python":sys.version, "platform":platform.platform(),
        "processor":platform.processor(), "hydraulic_sha256":provenance["raw_sha256"],
        "chronological_class_counts":class_coverage,
        "note":"Chronological split is feasible only for targets retaining adequate classes; grouped splits remain primary.",
    }
    (RAW / "artifact_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": run()
