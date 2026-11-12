"""Strict, serializable experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import math

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "torch_threads": 1,
    "data": {
        "source": "synthetic",
        "task": "smoke",
        "path": None,
        "input_dim": 16,
        "num_classes": 3,
        "train_samples": 90,
        "test_samples": 45,
    },
    "model": {
        "hidden_dim": 24,
        "neuron": "mdn",
        "timesteps": 12,
        "encoding": "rate",
        "threshold": 0.6,
        "current_gain": 3.0,
        "lif_decay": 0.8,
        "dt": 0.2,
        "tau": 1.0,
        "eta_a": 0.3,
        "eta_b": 0.25,
        "eta_c": 0.0,
        "eta_d": 0.1,
        "surrogate_width": 1.0,
    },
    "federation": {
        "clients": 5,
        "rounds": 8,
        "local_epochs": 1,
        "batch_size": 15,
        "learning_rate": 0.02,
        "dirichlet_alpha": 0.5,
        "min_client_samples": 2,
        "clip_grad_norm": 5.0,
        "offline": {},
        "flush_final": True,
    },
    "aggregation": {
        "policy": "event",
        "period": 1,
        "min_interval": 2,
        "max_interval": 4,
        "initial_threshold": 1.0,
        "beta": 1.0,
        "threshold_floor": 1e-8,
        "saliency_decay": 0.8,
        "keep_ratio": 0.25,
    },
    "network": {
        "one_way_ms": 3.0,
        "jitter_ms": 1.0,
        "drop_probability": 0.0,
        "rpc_timeout_ms": 80.0,
        "heartbeat_timeout_ms": 30.0,
        "control_message_bytes": 96,
    },
}


def _merge(base: dict, patch: dict, prefix: str = "") -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict) and key != "offline":
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            result[key] = _merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def validate(cfg: dict) -> None:
    d, m, f, a, n = (cfg[k] for k in ("data", "model", "federation", "aggregation", "network"))

    def positive_int(value: Any, key: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    for key in ("input_dim", "num_classes", "train_samples", "test_samples"):
        positive_int(d[key], f"data.{key}")
    for key in ("hidden_dim", "timesteps"):
        positive_int(m[key], f"model.{key}")
    for key in ("clients", "rounds", "local_epochs", "batch_size", "min_client_samples"):
        positive_int(f[key], f"federation.{key}")
    for key in ("period", "min_interval", "max_interval"):
        positive_int(a[key], f"aggregation.{key}")
    positive_int(cfg["torch_threads"], "torch_threads")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(f["flush_final"]) is not bool:
        raise ValueError("flush_final must be a boolean")
    for section_name, section in [("model", m), ("aggregation", a), ("network", n)]:
        for key, value in section.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{section_name}.{key} must be finite")
    for value, key in [
        (m["threshold"], "threshold"),
        (m["dt"], "dt"),
        (m["tau"], "tau"),
        (m["surrogate_width"], "surrogate_width"),
        (m["current_gain"], "current_gain"),
        (f["learning_rate"], "learning_rate"),
        (f["dirichlet_alpha"], "dirichlet_alpha"),
        (f["clip_grad_norm"], "clip_grad_norm"),
        (a["initial_threshold"], "initial_threshold"),
        (a["threshold_floor"], "threshold_floor"),
        (a["beta"], "beta"),
    ]:
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if d["source"] not in {"synthetic", "npz"}:
        raise ValueError("data.source must be synthetic or npz")
    if d["source"] == "npz" and not d["path"]:
        raise ValueError("NPZ data requires data.path")
    if d["num_classes"] < 2:
        raise ValueError("At least two classes are required")
    if m["neuron"] not in {"mdn", "lif"}:
        raise ValueError("model.neuron must be mdn or lif")
    if m["encoding"] not in {"rate", "latency", "phase", "burst", "preencoded"}:
        raise ValueError("Unknown spike encoding")
    if d["source"] == "synthetic" and m["encoding"] == "preencoded":
        raise ValueError("Synthetic examples are static features; choose a spike encoder")
    if not 0 <= m["lif_decay"] < 1:
        raise ValueError("lif_decay must be in [0, 1)")
    if a["policy"] not in {"event", "periodic"}:
        raise ValueError("aggregation.policy must be event or periodic")
    if not 0 < a["keep_ratio"] <= 1 or not 0 <= a["saliency_decay"] < 1:
        raise ValueError("Invalid keep_ratio or saliency_decay")
    if a["max_interval"] < a["min_interval"]:
        raise ValueError("max_interval must be >= min_interval")
    if not isinstance(f["offline"], dict):
        raise ValueError("federation.offline must map node IDs to round lists")
    for node, rounds in f["offline"].items():
        if str(int(node)) != str(node) or not 0 <= int(node) < f["clients"]:
            raise ValueError(f"Invalid offline node ID: {node}")
        if not isinstance(rounds, list) or any(type(r) is not int or r < 1 for r in rounds):
            raise ValueError("Offline rounds must be positive 1-based integers")
    if not 0 <= n["drop_probability"] < 1:
        raise ValueError("drop_probability must be in [0, 1)")
    if n["one_way_ms"] < 0 or n["jitter_ms"] < 0:
        raise ValueError("Network latencies must be nonnegative")
    if min(n["rpc_timeout_ms"], n["heartbeat_timeout_ms"]) <= 0:
        raise ValueError("Network timeouts must be positive")
    positive_int(n["control_message_bytes"], "control_message_bytes")


def configuration(patch: dict | None = None) -> dict:
    cfg = _merge(DEFAULTS, patch or {})
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        patch = yaml.safe_load(stream)
    if not isinstance(patch, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    cfg = configuration(patch)
    if cfg["data"]["source"] == "npz":
        data_path = Path(cfg["data"]["path"]).expanduser()
        cfg["data"]["path"] = str((path.parent / data_path).resolve())
    return cfg
