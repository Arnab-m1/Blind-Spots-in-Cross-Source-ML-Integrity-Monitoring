# Matched Hydraulic Security Protocol

Status: exploratory reanalysis with a frozen evaluation schema. It is not
described as predeclared.

## Common Cases

Every method is evaluated on the same split, test cycle, support, task, budget,
monitor, and alarm threshold. The support pool contains all 17 singleton
supports and 43 sensor pairs selected with seed 2027. The five existing split
seeds and budgets 0.10, 0.25, 0.50, 0.75, and 1.00 are retained.

The primary population contains test cycles whose clean task prediction is
correct and whose clean monitor score is below the validation-calibrated alarm
threshold. The complete test population is reported separately.

## Task Semantics

All four task scores use increasing values for increasing degradation. A
meaningful adverse event occurs when a clean correct prediction becomes an
incorrect prediction of a healthier physical condition. This excludes existing
task errors and arbitrary movement in the degradation direction.

The primary joint security outcome is a meaningful adverse event with no alarm
on the attacked cycle.

## Matched Methods

- `A0_untargeted_random`: fixed random parameter direction for each split and support.
- `A1_task_only_reassuring`: task-gradient direction toward a healthier report.
- `A2_generic_stealth_reassuring_sign`: least-visible monitor direction; only its sign is chosen toward a healthier report.
- `A4_task_conditioned_reassuring`: minimum-evidence direction for a fixed local shift toward a healthier report.

All methods use the same gain/scaled-bias parameterization and L2 budget.
Analytic directions are marked feasible only when their norm, gain, and finite
output checks pass.

## Saved Outputs

`matched_security_events.npz` stores per-cycle predictions, monitor scores, and
event indicators. CSV files provide split-level tables, budget curves, and
diagnostics. `matched_security_metadata.json` records array dimensions and
population counts.
