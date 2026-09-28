"""Build synchronized 1 Hz Hydraulic sequences without crossing cycles."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS

ARCHIVE = ROOT / "data/hydraulic/raw.zip"
OUT = ROOT / "data/hydraulic/processed/cycle_sequences.npz"


def run() -> None:
    channels = []
    with ZipFile(ARCHIVE) as archive:
        for sensor in SENSOR_SPECS:
            with archive.open(f"{sensor.name}.txt") as handle:
                values = np.loadtxt(handle, dtype=np.float32)
            expected = sensor.sampling_hz * 60
            if values.shape != (2205, expected):
                raise ValueError(f"unexpected shape for {sensor.name}: {values.shape}")
            channels.append(values.reshape(2205, 60, sensor.sampling_hz).mean(axis=2))
    sequences = np.stack(channels, axis=2).astype(np.float32)
    np.savez_compressed(OUT, sequences=sequences,
                        sensor_names=np.asarray([sensor.name for sensor in SENSOR_SPECS]))
    metadata = {"shape": list(sequences.shape), "seconds_per_cycle": 60,
                "synchronization": "one-second mean", "cycle_boundaries_preserved": True}
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
