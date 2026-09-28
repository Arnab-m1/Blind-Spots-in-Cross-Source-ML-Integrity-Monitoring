"""Runtime and failure benchmark for core certificate and attack routines."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from satml_monitoring.attacks import constrained_oracle_attack, continuation_attack
from satml_monitoring.certificates import task_visibility_certificate

OUTPUT = ROOT / "results" / "raw" / "benchmarks.json"


def timed(function, repeats):
    failures = 0; started = time.perf_counter()
    for _ in range(repeats):
        try: function()
        except (ValueError, ArithmeticError, RuntimeError): failures += 1
    elapsed = time.perf_counter() - started
    return {"repeats": repeats, "seconds": elapsed, "microseconds_per_call": 1e6 * elapsed / repeats, "failures": failures}


def run():
    rng = np.random.default_rng(2027)
    j = rng.normal(size=(8, 15)); u = rng.normal(size=15)
    monitor = lambda x: np.array([x[0] - x[1], x[2] + 0.1 * x[0] ** 2])
    task = lambda x: 2 * x[0] + x[1] - x[2]
    point = np.zeros(3)
    report = {
        "certificate": timed(lambda: task_visibility_certificate(j, u), 5000),
        "continuation": timed(lambda: continuation_attack(monitor, task, point, [0, 1, 2], 0.5, steps=10), 100),
        "oracle": timed(lambda: constrained_oracle_attack(monitor, task, point, [0, 1, 2], 0.5, 0.05), 50),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": run()
