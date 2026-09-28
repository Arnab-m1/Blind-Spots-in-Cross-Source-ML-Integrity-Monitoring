# External Monitor Protocol

Version: `2026-09-21-p1-v1`.

Each UCI Hydraulic cycle is a separate 60-second multivariate sequence. The
100 Hz and 10 Hz channels are averaged within each second; native 1 Hz channels
are unchanged. Windows have length ten and never cross cycle boundaries.
Training, validation, and test cycles use the frozen seed-17 grouped split.

GDN follows Deng and Hooi's forecasting design: learned sensor embeddings,
top-k embedding graph, graph attention, next-step forecasting, and robust
per-sensor deviation normalization. Its native point score is the maximum
normalized sensor deviation; the cycle alarm score is the maximum point score.

TranAD follows Tuli et al.'s transformer reconstruction design: sequence
encoding, focus-score self-conditioning, two reconstruction phases, and the
epoch-weighted adversarial objective. Its native point score is phase-two
per-feature squared reconstruction error; the cycle score is the maximum
feature-summed point score.

The positive control exposes the full standardized per-sensor forecast error.
It is first-order identifying for every audited singleton and pair support.

CE, CP, and SE are treated as independently reported telemetry channels. An
edit to a measured channel does not implicitly propagate to those channels;
an attacker must compromise a derived channel explicitly for it to change.
This avoids inventing undocumented trusted recomputation formulas.
