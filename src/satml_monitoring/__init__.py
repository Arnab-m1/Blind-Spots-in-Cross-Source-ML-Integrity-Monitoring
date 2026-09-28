"""Task-conditioned monitoring certificates, attacks, and defenses."""

from .attacks import (
    constrained_oracle_attack,
    continuation_attack,
    finite_difference_jacobian,
    minimum_evidence_attack,
    task_gradient_attack,
)
from .certificates import Certificate, task_visibility_certificate
from .geometry import harmful_nullspace_score, nullspace, numerical_rank
from .hardening import CandidateCheck, greedy_task_aware_hardening

__all__ = [
    "CandidateCheck",
    "Certificate",
    "constrained_oracle_attack",
    "continuation_attack",
    "finite_difference_jacobian",
    "greedy_task_aware_hardening",
    "harmful_nullspace_score",
    "minimum_evidence_attack",
    "nullspace",
    "numerical_rank",
    "task_gradient_attack",
    "task_visibility_certificate",
]
