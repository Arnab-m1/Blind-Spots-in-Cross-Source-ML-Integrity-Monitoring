"""Train and evaluate faithful temporal monitors on Hydraulic cycles."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS, TARGET_NAMES, newly_reassuring_error, summary_parameter_jacobian,
)
from satml_monitoring.sequences import cycle_windows as windows
from satml_monitoring.temporal_monitors import GDN, TranAD

CONFIG = ROOT / "configs/e6_external_monitors.json"
DATA = ROOT / "data/hydraulic/processed"
MODELS = ROOT / "results/models/e6_external_monitors"
RAW = ROOT / "results/raw/e6_external_monitors"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
SEVERITY_SIGNS = np.array([-1.0, -1.0, 1.0, -1.0])
TARGET_VALUES = (
    np.array([3, 20, 100]), np.array([73, 80, 90, 100]),
    np.array([0, 1, 2]), np.array([90, 100, 115, 130]),
)


def batches(values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = tuple(torch.from_numpy(value.astype(np.float32)) for value in values)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def train_gdn(model: GDN, train: np.ndarray, config: dict) -> list[float]:
    x, y = windows(train, config["window"], predict=True)
    loader = batches((x, y), config["models"]["gdn"]["batch_size"], True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["models"]["gdn"]["learning_rate"])
    history = []
    model.train()
    for _ in range(config["models"]["gdn"]["epochs"]):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(xb)
        history.append(total / len(x))
    return history


def train_tranad(model: TranAD, train: np.ndarray, config: dict) -> list[float]:
    x, _ = windows(train, config["window"], predict=False)
    loader = batches((x,), config["models"]["tranad"]["batch_size"], True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["models"]["tranad"]["learning_rate"])
    history = []
    model.train()
    epochs = config["models"]["tranad"]["epochs"]
    for epoch in range(epochs):
        total = 0.0
        weight = 1.0 / (epoch + 1)
        for (xb,) in loader:
            optimizer.zero_grad()
            first, second = model.reconstruct(xb)
            loss = weight * nn.functional.mse_loss(first, xb) + (1.0 - weight) * nn.functional.mse_loss(second, xb)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(xb)
        history.append(total / len(x))
    return history


@torch.no_grad()
def residuals(model: nn.Module, sequences: np.ndarray, kind: str, window: int,
              batch_size: int = 1024) -> np.ndarray:
    model.eval()
    if kind == "gdn":
        x, y = windows(sequences, window, predict=True)
        output = []
        for xb, yb in batches((x, y), batch_size, False):
            output.append((model(xb) - yb).cpu().numpy())
        return np.concatenate(output).reshape(len(sequences), sequences.shape[1] - window, -1)
    x, _ = windows(sequences, window, predict=False)
    output = []
    for (xb,) in batches((x,), batch_size, False):
        output.append((model(xb)[:, -1] - xb[:, -1]).cpu().numpy())
    return np.concatenate(output).reshape(len(sequences), sequences.shape[1] - window + 1, -1)


def score(residual: np.ndarray, kind: str, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = (residual - center) / scale
    point = np.max(np.abs(normalized), axis=2) if kind == "gdn" else np.sum(normalized ** 2, axis=2)
    return np.max(point, axis=1)


def sequence_summaries(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=1); std = sequence.std(axis=1)
    rms = np.sqrt(np.mean(sequence ** 2, axis=1))
    return np.stack([mean, std, rms, sequence.min(axis=1), sequence.max(axis=1)], axis=2).reshape(len(sequence), -1)


def parameter_columns(support: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([np.arange(2 * index, 2 * index + 2) for index in support])


def support_name(support: tuple[int, ...]) -> str:
    return "+".join(SENSOR_SPECS[index].name for index in support)


def unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else value


def nearest(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmin(np.abs(values[:, None] - classes[None, :]), axis=1)]


def apply_attack(sequence: np.ndarray, theta: np.ndarray, bias_scale: np.ndarray) -> np.ndarray:
    gain = theta[0::2]; bias = theta[1::2] * bias_scale
    return sequence * (1.0 + gain[None, None, :]) + bias[None, None, :]


def finite_jacobian(model: nn.Module, kind: str, base: np.ndarray, bias_scale: np.ndarray,
                    sequence_mean: np.ndarray, sequence_scale: np.ndarray,
                    scale: np.ndarray, window: int) -> np.ndarray:
    """Differentiate the full signed native residual map by central differences."""
    step = 1e-4
    columns = []
    for index in range(2 * len(SENSOR_SPECS)):
        offset = np.zeros(2 * len(SENSOR_SPECS)); offset[index] = step
        plus_sequence = (apply_attack(base, offset, bias_scale) - sequence_mean) / sequence_scale
        minus_sequence = (apply_attack(base, -offset, bias_scale) - sequence_mean) / sequence_scale
        plus = residuals(model, plus_sequence, kind, window)
        minus = residuals(model, minus_sequence, kind, window)
        columns.append((((plus - minus) / (2 * step)) / scale).ravel())
    return np.column_stack(columns)


def selected_supports(config: dict) -> list[tuple[int, ...]]:
    singles = [(index,) for index in range(len(SENSOR_SPECS))]
    pairs = list(itertools.combinations(range(len(SENSOR_SPECS)), 2))
    rng = np.random.default_rng(config["support_sampling_seed"])
    selected = rng.choice(len(pairs), config["support_pair_count"], replace=False)
    return singles + [pairs[int(index)] for index in selected]


def evaluate_monitor(name: str, model: nn.Module, sequence: np.ndarray, splits, config,
                     sequence_mean, sequence_scale, bias_scale, fit, targets) -> tuple[list[dict], list[dict], dict]:
    train = splits["train"]; validation = splits["validation"]; test = splits["test"]
    z = (sequence - sequence_mean) / sequence_scale
    train_residual = residuals(model, z[train], name, config["window"])
    center = np.median(train_residual, axis=(0, 1), keepdims=True)
    robust = np.quantile(train_residual, .75, axis=(0, 1), keepdims=True) - np.quantile(train_residual, .25, axis=(0, 1), keepdims=True)
    robust = np.maximum(robust, 1e-4)
    validation_score = score(residuals(model, z[validation], name, config["window"]), name, center, robust)
    threshold = float(np.quantile(validation_score, config["alarm_quantile"]))
    test_score = score(residuals(model, z[test], name, config["window"]), name, center, robust)

    base = np.mean(sequence[train], axis=0, keepdims=True)
    monitor_j = finite_jacobian(model, name, base, bias_scale, sequence_mean,
                                sequence_scale, robust, config["window"])
    # Full signed residuals are the positive visibility control.
    full_j = np.column_stack([
        ((apply_attack(base, np.eye(34)[column], bias_scale) - base) / sequence_scale).ravel()
        for column in range(34)
    ])
    summaries = sequence_summaries(sequence)
    physical_j = summary_parameter_jacobian(np.mean(summaries[train], axis=0, keepdims=True), bias_scale)[0]
    task_j = physical_j / fit["scaler"].scale_[:, None]
    gradients = fit["linear"].coef_ @ task_j
    supports = selected_supports(config)
    certificate_rows = []
    event_rows = []
    rng = np.random.default_rng(config["support_sampling_seed"])
    chosen_test = np.sort(rng.choice(test, min(config["evaluation_cycles"], len(test)), replace=False))
    clean_z = fit["scaler"].transform(summaries[chosen_test])
    clean_prediction = fit["linear"].predict(clean_z)
    true_standard = fit["target_scaler"].transform(targets[chosen_test, :4] * SEVERITY_SIGNS)
    clean_alarm = test_score[np.searchsorted(test, chosen_test)] > threshold

    for support in supports:
        columns = parameter_columns(support)
        for task_index, task in enumerate(TARGET_NAMES[:4]):
            gradient = gradients[task_index, columns]
            certificate = task_visibility_certificate(monitor_j[:, columns], gradient)
            full_certificate = task_visibility_certificate(full_j[:, columns], gradient)
            certificate_rows.append({
                "monitor": name, "support": support_name(support), "support_size": len(support),
                "task": task, "rank": certificate.rank, "nullity": certificate.nullity,
                "gamma": certificate.gamma, "is_blind": certificate.is_blind,
                "full_residual_rank": full_certificate.rank,
                "full_residual_gamma": full_certificate.gamma,
            })
            _, _, vh = np.linalg.svd(monitor_j[:, columns], full_matrices=True)
            generic = -np.sign(vh[-1] @ gradient) * unit(vh[-1])
            conditioned = unit(minimum_evidence_attack(monitor_j[:, columns], gradient, -0.25))
            for attack_name, direction in (("A2_generic_stealth", generic), ("A4_task_conditioned", conditioned)):
                for budget in config["budgets"]:
                    theta = np.zeros(34); theta[columns] = budget * direction
                    attacked = apply_attack(sequence[chosen_test], theta, bias_scale)
                    attacked_summary = sequence_summaries(attacked)
                    prediction = fit["linear"].predict(fit["scaler"].transform(attacked_summary))[:, task_index]
                    attacked_z = (attacked - sequence_mean) / sequence_scale
                    alarm = score(residuals(model, attacked_z, name, config["window"]), name, center, robust) > threshold
                    classes = fit["target_scaler"].transform(
                        np.tile(np.mean(targets[train, :4] * SEVERITY_SIGNS, axis=0), (len(TARGET_VALUES[task_index]), 1))
                    )[:, task_index]
                    physical = TARGET_VALUES[task_index] * SEVERITY_SIGNS[task_index]
                    probe = np.tile(np.mean(targets[train, :4] * SEVERITY_SIGNS, axis=0), (len(physical), 1)); probe[:, task_index] = physical
                    classes = fit["target_scaler"].transform(probe)[:, task_index]
                    clean_class = nearest(clean_prediction[:, task_index], classes)
                    true_class = nearest(true_standard[:, task_index], classes)
                    attacked_class = nearest(prediction, classes)
                    adverse = newly_reassuring_error(clean_class, attacked_class, true_class)
                    primary = (clean_class == true_class) & ~clean_alarm
                    denominator = max(int(primary.sum()), 1)
                    event_rows.append({
                        "monitor": name, "support": support_name(support), "support_size": len(support),
                        "task": task, "attack": attack_name, "budget": budget,
                        "population": int(primary.sum()), "gamma": certificate.gamma,
                        "meaningful_adverse_rate": float(np.sum(adverse & primary) / denominator),
                        "alarm_rate": float(np.sum(alarm & primary) / denominator),
                        "joint_adverse_no_alarm_rate": float(np.sum(adverse & ~alarm & primary) / denominator),
                        "median_task_movement": float(np.median((prediction - clean_prediction[:, task_index])[primary])) if np.any(primary) else np.nan,
                    })
    metadata = {
        "monitor": name, "validation_threshold": threshold,
        "validation_alarm_rate": float(np.mean(validation_score > threshold)),
        "test_alarm_rate": float(np.mean(test_score > threshold)),
        "evaluation_test_cycles": chosen_test.tolist(),
        "residual_center": center.ravel().tolist(), "residual_scale": robust.ravel().tolist(),
    }
    return certificate_rows, event_rows, metadata


def prioritization(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditioned = events[(events.attack == "A4_task_conditioned") & (events.budget == events.budget.max())]
    for monitor, frame in conditioned.groupby("monitor"):
        ranked = frame.sort_values("gamma", ascending=True).reset_index(drop=True)
        positives = ranked.joint_adverse_no_alarm_rate > 0
        for k in (5, 10, 20):
            top = positives.iloc[:min(k, len(positives))]
            rows.append({"monitor": monitor, "k": k, "precision_at_k": float(top.mean()),
                         "failures_found": int(top.sum()), "total_failures": int(positives.sum())})
    return pd.DataFrame(rows)


def run() -> None:
    started = time.perf_counter()
    torch.manual_seed(17); np.random.seed(17); torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    config = json.loads(CONFIG.read_text())
    for path in (MODELS, RAW, TABLES, FIGURES): path.mkdir(parents=True, exist_ok=True)
    sequence = np.load(DATA / "cycle_sequences.npz")["sequences"].astype(np.float32)
    split_data = np.load(DATA / "grouped_splits.npz")
    seed = config["split_seed"]
    splits = {part: split_data[f"seed_{seed}_{part}"] for part in ("train", "validation", "test")}
    feature_data = np.load(DATA / "cycle_features.npz")
    targets = feature_data["targets"].astype(float)
    summaries = feature_data["features"].astype(float)
    severity_targets = targets[:, :4] * SEVERITY_SIGNS
    task_scaler = StandardScaler().fit(summaries[splits["train"]])
    target_scaler = StandardScaler().fit(severity_targets[splits["train"]])
    linear = Ridge(alpha=10.0).fit(
        task_scaler.transform(summaries[splits["train"]]),
        target_scaler.transform(severity_targets[splits["train"]]),
    )
    fit = {"scaler": task_scaler, "target_scaler": target_scaler, "linear": linear}
    sequence_mean = sequence[splits["train"]].mean(axis=(0, 1), keepdims=True)
    sequence_scale = np.maximum(sequence[splits["train"]].std(axis=(0, 1), keepdims=True), 1e-6)
    z_train = (sequence[splits["train"]] - sequence_mean) / sequence_scale
    bias_scale = np.maximum(sequence[splits["train"]].mean(axis=1).std(axis=0), 1e-8)

    gcfg = config["models"]["gdn"]
    tcfg = config["models"]["tranad"]
    models = {
        "gdn": GDN(17, config["window"], gcfg["embedding_dim"], gcfg["hidden_dim"], gcfg["top_k"]),
        "tranad": TranAD(17, config["window"], tcfg["model_dim"], tcfg["heads"], tcfg["layers"]),
    }
    histories = {"gdn": train_gdn(models["gdn"], z_train, config)}
    histories["tranad"] = train_tranad(models["tranad"], z_train, config)
    certificate_rows = []; event_rows = []; monitor_metadata = {}
    for name, model in models.items():
        torch.save(model.state_dict(), MODELS / f"{name}_seed_{seed}.pt")
        certificates, events, metadata = evaluate_monitor(
            name, model, sequence, splits, config, sequence_mean, sequence_scale,
            bias_scale, fit, targets,
        )
        certificate_rows.extend(certificates); event_rows.extend(events); monitor_metadata[name] = metadata
    certificates = pd.DataFrame(certificate_rows)
    events = pd.DataFrame(event_rows)
    priority = prioritization(events)
    certificates.to_csv(RAW / "certificates.csv", index=False)
    events.to_csv(RAW / "matched_security.csv", index=False)
    priority.to_csv(RAW / "prioritization.csv", index=False)
    summary = events.groupby(["monitor", "attack", "budget"], as_index=False).agg(
        median_adverse=("meaningful_adverse_rate", "median"), median_alarm=("alarm_rate", "median"),
        median_joint=("joint_adverse_no_alarm_rate", "median"), max_joint=("joint_adverse_no_alarm_rate", "max"),
    )
    summary.to_csv(TABLES / "table_external_monitors.csv", index=False)
    metadata = {
        "protocol": config, "training_loss": histories, "monitors": monitor_metadata,
        "sequence_mean": sequence_mean.ravel().tolist(), "sequence_scale": sequence_scale.ravel().tolist(),
        "bias_scale": bias_scale.tolist(), "runtime_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__, "device": "cpu",
    }
    (RAW / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False)); print(priority.to_string(index=False))


if __name__ == "__main__":
    run()
