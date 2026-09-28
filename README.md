# Task-Conditioned Monitor Auditing Artifact

This artifact contains the implementation, datasets, frozen outputs, and tests
for the paper *Task-Conditioned Blind Spots in Cross-Source ML Integrity
Monitoring*.

## Contents

- `src/`: certificates, attacks, model adapters, dataset utilities, and defenses.
- `experiments/`: synthetic, ICRISAT, Hydraulic, gray-box, and hardening runs.
- `data/`: dataset files, processed Hydraulic features, splits, and provenance.
- `results/`: frozen raw rows, tables, figures, and fitted Hydraulic models.
- `tests/`: algebraic, numerical, preprocessing, adapter, and schema tests.
- `reproduce.sh`: full experiment and audit pipeline.

## Environment

Use Python 3.10 or newer. A fresh environment can be prepared with:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install pytest
```

## Fast Verification

```bash
python -m pytest -q
python experiments/audit.py
```

The artifact test suite contains 25 tests. The audit checks result schemas, required
figures, split coverage, and the Hydraulic archive hash.

## Full Reproduction

```bash
./reproduce.sh
```

The pipeline regenerates raw results before tables and figures. Outputs are
written under `results/`. The nonlinear certificate and stability experiments
account for most of the runtime. The external temporal-monitor experiment uses
the pinned PyTorch dependency, trains on CPU by default, preserves Hydraulic
cycle boundaries, and documents its adaptations in
`experiments/e6_external_monitors/PROTOCOL.md`.

## Paper Result Map

| Result | Generator | Output |
|---|---|---|
| Synthetic geometry and rates | `experiments/e0_synthetic/` | `results/raw/e0_synthetic/` |
| ICRISAT linear audit | `experiments/e1_icrisat_linear/run.py` | `results/raw/e1_icrisat_linear/` |
| ICRISAT nonlinear and frontier audit | `experiments/e2_icrisat_nonlinear/` | `results/raw/e2_icrisat_nonlinear/` |
| Hydraulic models and attacks | `experiments/e3_hydraulic/` | `results/raw/e3_hydraulic/` |
| Gray-box transfer | `experiments/e4_graybox/run.py` | `results/raw/e4_graybox/` |
| Adaptive hardening | `experiments/e5_hardening/` | `results/raw/e5_hardening/` |
| GDN and TranAD validation | `experiments/e6_external_monitors/` | `results/raw/e6_external_monitors/` |
| Held-out defender decision | `experiments/e7_defender_decision/` | `results/raw/e7_defender_decision/` |
| Whitening and scalar-score robustness | `experiments/e8_metric_robustness/` | `results/raw/e8_metric_robustness/` |
| Exactly blind larger-support attacks | `experiments/e9_blind_larger_supports/` | `results/raw/e9_blind_larger_supports/` |
| ICRISAT plausibility audit | `experiments/e10_plausibility/` | `results/raw/e10_plausibility/` |
| Tables and intervals | `experiments/statistical_report.py` | `results/tables/` |

## Data

`data/icrisat/PROVENANCE.json` records the ICRISAT source DOI, CC BY 4.0
license, hashes, and file roles. `data/hydraulic/PROVENANCE.json` records the
UCI source DOI, CC BY 4.0 license, archive hash, and cycle inventory.

All paths are relative to the artifact root. Runs use fixed seeds and preserve
per-example or per-support rows for independent analysis.
