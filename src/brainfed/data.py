"""Offline synthetic smoke data and explicit train/test NPZ adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class DataBundle:
    train: TensorDataset
    test: TensorDataset
    provenance: dict


def load_data(cfg: dict) -> DataBundle:
    data = cfg["data"]
    if data["source"] == "synthetic":
        rng = np.random.default_rng(cfg["seed"] + 101)
        prototypes = rng.uniform(0.1, 0.9, (data["num_classes"], data["input_dim"]))

        def split(count: int):
            labels = np.arange(count, dtype=np.int64) % data["num_classes"]
            rng.shuffle(labels)
            features = np.clip(
                prototypes[labels] + rng.normal(0, 0.07, (count, data["input_dim"])), 0, 1
            ).astype(np.float32)
            return features, labels

        x_train, y_train = split(data["train_samples"])
        x_test, y_test = split(data["test_samples"])
        provenance = {
            "source": "synthetic",
            "real_sensor_data": False,
            "generator_seed": cfg["seed"] + 101,
        }
    else:
        path = Path(data["path"])
        with np.load(path, allow_pickle=False) as arrays:
            keys = ("x_train", "y_train", "x_test", "y_test")
            missing = set(keys) - set(arrays.files)
            if missing:
                raise ValueError(f"NPZ is missing keys: {sorted(missing)}")
            x_train, y_train, x_test, y_test = (np.asarray(arrays[k]) for k in keys)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance = {
            "source": "npz",
            "filename": path.name,
            "sha256": digest.hexdigest(),
            "real_sensor_data": "user-provided; verify dataset provenance",
        }
    train = _dataset(x_train, y_train, cfg, "train")
    test = _dataset(x_test, y_test, cfg, "test")
    if train.tensors[0].ndim != test.tensors[0].ndim:
        raise ValueError("Train/test feature ranks must match")
    provenance.update(
        {
            "train_samples": len(train),
            "test_samples": len(test),
            "input_dim": data["input_dim"],
            "num_classes": data["num_classes"],
        }
    )
    return DataBundle(train, test, provenance)


def _dataset(x: np.ndarray, y: np.ndarray, cfg: dict, name: str) -> TensorDataset:
    d, m = cfg["data"], cfg["model"]
    if x.ndim not in (2, 3) or x.shape[-1] != d["input_dim"] or len(x) == 0:
        raise ValueError(f"{name}: invalid feature shape {x.shape}")
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{name}: labels must be a one-dimensional integer array")
    if y.min() < 0 or y.max() >= d["num_classes"]:
        raise ValueError(f"{name}: label outside configured class range")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(f"{name}: features must be finite values in [0, 1]")
    if m["encoding"] == "preencoded":
        if x.ndim != 3 or x.shape[1] != m["timesteps"] or not np.isin(x, [0, 1]).all():
            raise ValueError(f"{name}: preencoded input must be binary [N,T,F]")
    elif x.ndim != 2:
        raise ValueError(f"{name}: static encoder expects [N,F]")
    return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))


def dirichlet_partition(
    labels, clients: int, alpha: float, seed: int, min_size: int = 2
) -> list[list[int]]:
    labels = np.asarray(labels)
    if clients < 1 or alpha <= 0 or min_size < 1 or len(labels) < clients * min_size:
        raise ValueError("Not enough samples or invalid non-IID partition settings")
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        shards: list[list[int]] = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            cuts = (np.cumsum(rng.dirichlet(np.full(clients, alpha)))[:-1] * len(indices)).astype(
                int
            )
            for shard, values in zip(shards, np.split(indices, cuts)):
                shard.extend(int(i) for i in values)
        if min(map(len, shards)) >= min_size:
            for shard in shards:
                rng.shuffle(shard)
            return shards
    raise ValueError("Cannot obtain requested partition; increase alpha or sample count")
