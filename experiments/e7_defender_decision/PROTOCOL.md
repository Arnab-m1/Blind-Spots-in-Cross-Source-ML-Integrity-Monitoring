# Held-Out Defender Decision Protocol

Version: `2026-09-21-p2-v1`.

The experiment uses the support-group split and independent finite labels from
P0-C. Source checks are selected using development-support cases only. Final
risk is measured only on evaluation-support cases. Each protected source adds
direct integrity checks for its gain and scaled-bias controls.

Five selectors receive the same source budget: seeded random order, monitor
conditioning, task-gradient magnitude, greedy local margin, and greedy coverage
of development cases with independently verified finite failures. The latter
is the finite-audit-guided method.

For every held-out case, protection removes the selected source controls from
the admissible attack support. Previously negative cases remain negative because
the feasible set only shrinks. Previously positive cases with unchanged support
retain their label. Every positive case whose support intersects a protected
source is re-run with the same native alarm, attack budget, and multi-start
finite optimizer. Empty residual supports are secure by construction.

The primary outcome is residual held-out verified joint failures. The experiment
does not replace P0-C audit prioritization as the primary defender decision and
does not assume that a local-margin selector must minimize finite risk.
