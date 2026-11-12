"""Event timing and neuron-row sparse model-delta aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def clone_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_bytes(state: dict[str, Tensor]) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


class SaliencyTracker:
    """EMA of nonnegative per-neuron update magnitude, summed over parameters.

    Weight-row L2 norm and absolute bias update contribute to each neuron's
    score. Signed coordinate updates are not summed, avoiding cancellation.
    """

    def __init__(self, decay: float = 0.8):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.scores: dict[str, Tensor] = {}

    def update(self, before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
        layer_scores: dict[str, Tensor] = {}
        for key, old in before.items():
            delta = (after[key] - old).detach().cpu()
            layer, _ = key.rsplit(".", 1)
            magnitude = delta.reshape(delta.shape[0], -1).norm(dim=1)
            layer_scores[layer] = layer_scores.get(layer, torch.zeros_like(magnitude)) + magnitude
        for layer, magnitude in layer_scores.items():
            previous = self.scores.get(layer, torch.zeros_like(magnitude))
            self.scores[layer] = self.decay * previous + (1 - self.decay) * magnitude
        return sum(score.sum().item() for score in self.scores.values())


class AggregationClock:
    """Round-level communication opportunities, distinct from neural time steps.

    The adaptive threshold uses completed events only. The current event's
    accumulated score is added to history after a successful aggregation.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.elapsed = 0
        self.accumulated = 0.0
        self.threshold = config["initial_threshold"]
        self.history: list[float] = []

    def observe(self, score: float) -> bool:
        if not math.isfinite(score) or score < 0:
            raise ValueError("Saliency must be finite and nonnegative")
        self.elapsed += 1
        self.accumulated += score
        if self.cfg["policy"] == "periodic":
            return self.elapsed >= self.cfg["period"]
        return self.elapsed >= self.cfg["max_interval"] or (
            self.elapsed >= self.cfg["min_interval"] and self.accumulated >= self.threshold
        )

    def committed(self) -> None:
        self.history.append(self.accumulated)
        self.threshold = max(
            self.cfg["threshold_floor"], self.cfg["beta"] * sum(self.history) / len(self.history)
        )
        self.elapsed, self.accumulated = 0, 0.0


@dataclass
class SparseTensor:
    indices: Tensor
    values: Tensor
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return self.indices.numel() * self.indices.element_size() + (
            self.values.numel() * self.values.element_size()
        )


@dataclass
class Update:
    client: int
    samples: int
    base_version: int
    tensors: dict[str, Tensor | SparseTensor]

    @property
    def nbytes(self) -> int:
        return sum(
            t.nbytes if isinstance(t, SparseTensor) else t.numel() * t.element_size()
            for t in self.tensors.values()
        )


def pack_update(
    client: int,
    samples: int,
    version: int,
    local: dict[str, Tensor],
    global_state: dict[str, Tensor],
    tracker: SaliencyTracker,
    keep_ratio: float,
) -> Update:
    if not 0 < keep_ratio <= 1 or samples <= 0:
        raise ValueError("Invalid keep ratio or client sample count")
    selected = {}
    if keep_ratio < 1:
        for layer, score in tracker.scores.items():
            k = max(1, math.ceil(len(score) * keep_ratio))
            selected[layer] = torch.topk(score, k, sorted=True).indices
    tensors: dict[str, Tensor | SparseTensor] = {}
    for key, baseline in global_state.items():
        delta = (local[key].detach().cpu() - baseline).contiguous()
        if keep_ratio == 1:
            tensors[key] = delta
        else:
            layer, _ = key.rsplit(".", 1)
            if layer not in selected:
                raise ValueError(f"Saliency missing for layer {layer}")
            row_size = delta[0].numel()
            indices = (selected[layer][:, None] * row_size + torch.arange(row_size)[None]).reshape(
                -1
            )
            tensors[key] = SparseTensor(
                indices.to(torch.int64), delta.flatten()[indices].clone(), tuple(delta.shape)
            )
    return Update(client, samples, version, tensors)


def aggregate(global_state: dict[str, Tensor], updates: list[Update], version: int):
    """Sample-weighted delta average; omitted coordinates contribute zero.

    At keep_ratio=1 this equals ordinary sample-weighted FedAvg of the
    participating clients. Sparse weights normalize over ALL participating
    samples, not just clients that happened to transmit a coordinate.
    """
    if not updates or len({u.client for u in updates}) != len(updates):
        raise ValueError("A nonempty set of distinct client updates is required")
    if any(u.base_version != version or u.samples <= 0 for u in updates):
        raise ValueError("Stale version or invalid sample count")
    result = clone_state(global_state)
    total = sum(u.samples for u in updates)
    for update in updates:
        if set(update.tensors) != set(global_state):
            raise ValueError("Update schema differs from global model")
        weight = update.samples / total
        for key, value in update.tensors.items():
            target = result[key]
            if isinstance(value, SparseTensor):
                if value.shape != tuple(target.shape) or value.indices.ndim != 1:
                    raise ValueError("Sparse update shape mismatch")
                if value.values.ndim != 1 or value.indices.numel() != value.values.numel():
                    raise ValueError("Sparse index/value size mismatch")
                if value.indices.dtype != torch.int64:
                    raise ValueError("Sparse indices must be int64")
                if value.indices.numel() and (
                    value.indices.min() < 0 or value.indices.max() >= target.numel()
                ):
                    raise ValueError("Sparse coordinate out of range")
                if value.indices.unique().numel() != value.indices.numel():
                    raise ValueError("Duplicate sparse coordinates")
                if not torch.isfinite(value.values).all():
                    raise ValueError("Nonfinite update")
                target.view(-1).index_add_(0, value.indices, value.values * weight)
            else:
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError("Dense update shape mismatch or nonfinite update")
                target.add_(value, alpha=weight)
    return result
