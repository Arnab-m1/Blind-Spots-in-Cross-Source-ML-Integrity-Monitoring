from __future__ import annotations
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
class ResultSchemaTests(unittest.TestCase):
    def test_matched_security_event_invariants(self):
        path = ROOT / "results/raw/e3_hydraulic/matched_security_events.npz"
        events = np.load(path)
        expected_shape = (5, 60, 4, 5, 4, 440)
        self.assertEqual(events["joint_adverse_no_alarm"].shape, expected_shape)
        np.testing.assert_array_equal(
            events["joint_adverse_no_alarm"],
            events["meaningful_adverse"] & ~events["alarm"],
        )
        for seed_position, valid in enumerate(events["valid_state"]):
            invalid = ~valid
            self.assertTrue(
                np.all(np.isnan(events["attacked_prediction"][seed_position, ..., invalid]))
            )

    def test_matched_security_protocol_is_frozen(self):
        config = json.loads(
            (ROOT / "configs/e3_hydraulic_matched.json").read_text()
        )
        self.assertEqual(config["support_design"]["singleton_supports"], "all_17")
        self.assertEqual(config["support_design"]["pair_supports"], 43)
        self.assertEqual(config["primary_population"], "clean_task_correct_and_clean_monitor_accepted")
        self.assertIsNone(config["continuous_threshold_eta"])

    def test_major_result_schemas(self):
        cases = {
            "results/raw/e1_icrisat_linear/support_certificates.csv":{"support","gamma","h"},
            "results/raw/e1_icrisat_linear/monitor_variants.csv":{"dimension","vector_gamma","scalar_blind_fraction"},
            "results/raw/e2_icrisat_nonlinear/attacks.csv":{"attack","task_shift","monitor_shift"},
            "results/raw/e2_icrisat_nonlinear/seed_stability.csv":{"seed","r2","gamma_a4_harm_spearman"},
            "results/raw/e2_icrisat_nonlinear/frontier_calibration.csv":{"delta","gamma","frontier_evidence","success"},
            "results/raw/e3_hydraulic/compression_ablation.csv":{"dimension","test_fpr","blind_fraction"},
            "results/raw/e3_hydraulic/attack_results.csv":{"attack","class_flip_rate","false_negative_rate","perturbation_norm"},
            "results/raw/e3_hydraulic/seed_stability_summary.csv":{"seed","attack","median_abs_task_shift","median_detection","blind_fraction"},
            "results/raw/e3_hydraulic/task_metric_example.csv":{"task","rank","sigma_min","gamma","median_abs_task_shift"},
            "results/raw/e3_hydraulic/matched_security_summary.csv":{"seed","support","task","budget","attack","population","meaningful_adverse_rate","alarm_rate","joint_adverse_no_alarm_rate","feasible"},
            "results/raw/e3_hydraulic/matched_security_diagnostics.csv":{"seed","support","task","budget","attack","constraint_residual","solver_status","feasible"},
            "results/raw/e3_hydraulic/matched_security_main_table.csv":{"seed","attack","population","clean_task_accuracy","clean_alarm_rate","meaningful_adverse_rate","alarm_rate","joint_adverse_no_alarm_rate","invalid_rate"},
            "results/raw/e3_hydraulic/matched_security_factor_summary.csv":{"seed","attack","task","support_family","budget","population","finite_nonlinearity_gap"},
            "results/raw/e3_hydraulic/matched_security_condition_summary.csv":{"seed","attack","task","condition_label","support_family","budget","population","joint_adverse_no_alarm_rate"},
            "results/raw/e3_hydraulic/prioritization_cases.csv":{"case_id","partition","gamma","verified_failure"},
            "results/raw/e3_hydraulic/prioritization_verification.csv":{"case_id","budget","feasible","verified_failure","constraint_residual"},
            "results/raw/e3_hydraulic/prioritization_metrics.csv":{"seed","method","budget","precision","recall","audc","wall_seconds"},
            "results/raw/e4_graybox/graybox_results.csv":{"setting","epsilon_j","epsilon_f"},
            "results/raw/e5_hardening/hardening_results.csv":{"method","budget","adaptive_unit_budget_harm"},
            "results/raw/e5_hardening/hydraulic_hardening_results.csv":{"method","budget","blind_task_supports","adaptive_unit_budget_harm"},
            "results/raw/e6_external_monitors/certificates.csv":{"monitor","support","task","gamma","rank","full_residual_rank"},
            "results/raw/e6_external_monitors/matched_security.csv":{"monitor","support","task","attack","budget","joint_adverse_no_alarm_rate"},
            "results/raw/e6_external_monitors/prioritization.csv":{"monitor","k","precision_at_k","failures_found"},
            "results/raw/e6_external_monitors/stability_certificates.csv":{"seed","monitor","support","task","gamma","is_blind"},
            "results/raw/e6_external_monitors/stability_matched_security.csv":{"seed","monitor","support","task","attack","budget","joint_adverse_no_alarm_rate"},
            "results/raw/e7_defender_decision/heldout_hardening.csv":{"seed","method","budget","original_failures","residual_failures","residual_joint_risk"},
            "results/raw/e7_defender_decision/finite_reruns.csv":{"case_id","method","protection_budget","verified_failure","feasible"},
            "results/raw/e8_metric_robustness/ranking_results.csv":{"seed","metric","budget","failures_found","precision"},
            "results/raw/e8_metric_robustness/paired_intervals.csv":{"budget","reference","comparator","paired_mean_difference","ci_low","ci_high"},
            "results/raw/e9_blind_larger_supports/selected_blind_configurations.csv":{"support_size","support","task","gamma","is_blind"},
            "results/raw/e9_blind_larger_supports/matched_attacks.csv":{"support_size","attack","budget_design","joint_adverse_no_alarm_rate"},
            "results/raw/e10_plausibility/support_budget_plausibility.csv":{"attack","budget","support","fraction_rows_nonnegative_physical_fields","fraction_rows_in_training_range"},
        }
        for relative, expected in cases.items():
            path=ROOT/relative; self.assertTrue(path.exists(), relative)
            self.assertTrue(expected <= set(pd.read_csv(path,nrows=1).columns), relative)

    def test_heldout_hardening_invariants(self):
        results = pd.read_csv(ROOT / "results/raw/e7_defender_decision/heldout_hardening.csv")
        self.assertEqual(set(results.method), {
            "random", "conditioning", "task_gradient", "local_margin", "finite_audit_guided"
        })
        self.assertTrue((results.residual_failures <= results.original_failures).all())
        baseline = results[results.budget == 0]
        self.assertTrue((baseline.residual_failures == baseline.original_failures).all())
        reruns = pd.read_csv(ROOT / "results/raw/e7_defender_decision/finite_reruns.csv")
        self.assertTrue(reruns.feasible.all())

    def test_operational_robustness_invariants(self):
        external = pd.read_csv(
            ROOT / "results/raw/e6_external_monitors/stability_certificates.csv"
        )
        self.assertEqual(set(external.seed), {17, 29, 43, 71})
        self.assertFalse(external.is_blind.any())
        blind = pd.read_csv(
            ROOT / "results/raw/e9_blind_larger_supports/selected_blind_configurations.csv"
        )
        self.assertTrue(blind.is_blind.all())
        self.assertEqual(set(blind.support_size), {3, 4})
        plausibility = pd.read_csv(
            ROOT / "results/raw/e10_plausibility/support_budget_plausibility.csv"
        )
        for column in (
            "fraction_rows_nonnegative_physical_fields",
            "fraction_rows_in_training_range",
        ):
            self.assertTrue(plausibility[column].between(0, 1).all())
if __name__ == "__main__": unittest.main()
