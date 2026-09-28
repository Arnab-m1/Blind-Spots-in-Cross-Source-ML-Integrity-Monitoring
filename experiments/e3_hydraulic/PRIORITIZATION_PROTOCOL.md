# Fixed-Budget Audit Prioritization Protocol

Version: `2026-09-21-p0c-v1`

The candidate unit is one `(split, state, support, task)` case. Each of the 60
frozen matched supports contributes one deterministically selected test state
per task when that state is initially classified correctly and accepted by the
native PCA-8 monitor. Supports whose stable SHA-256 index is congruent to zero
modulo three form the development partition; all other support groups form the
held-out evaluation partition. Model seeds occur in both partitions and are
not the partitioning unit.

A case is a verified failure only when a finite SLSQP search at budget 0.1,
0.25, or 0.5 returns a supported, norm-feasible manipulation that changes the
prediction toward a falsely reassuring physical class while remaining below
the validation-calibrated native alarm threshold. Failed or infeasible returns
are retained and reported, never counted as security successes.

Rankers are random order, rank/nullity, harmful-kernel score, task-gradient
magnitude, minimum whitened singular value, clean alarm slack, the
task-conditioned margin, and a nonnegative combined score. Combined-score
weights and all tie rules are selected on development supports only. Final
precision, recall, discovery area, and measured screening-plus-verification
time are computed on held-out support groups.
