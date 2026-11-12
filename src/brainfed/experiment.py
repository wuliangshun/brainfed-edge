"""Single-process federated-training orchestration with auditable metrics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import json
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from .aggregation import (
    AggregationClock,
    SaliencyTracker,
    aggregate,
    clone_state,
    pack_update,
    state_bytes,
)
from .config import validate
from .data import dirichlet_partition, load_data
from .election import SponsorElection
from .encoding import encode
from .model import build_model, rate_targets
from .network import SimulatedNetwork


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def seed_runtime(cfg: dict) -> torch.device:
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["torch_threads"])
    device = torch.device(cfg["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This prototype supports cpu or cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; set device: cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def evaluate(model: nn.Module, dataset, cfg: dict, device=None) -> dict:
    device = device or next(model.parameters()).device
    previous = model.training
    model.eval()
    classes = cfg["data"]["num_classes"]
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    total_loss = 0.0
    generator = torch.Generator().manual_seed(cfg["seed"] + 10000)
    x_all, y_all = dataset.tensors
    with torch.no_grad():
        for start in range(0, len(dataset), cfg["federation"]["batch_size"]):
            x, y = (
                x_all[start : start + cfg["federation"]["batch_size"]],
                y_all[start : start + cfg["federation"]["batch_size"]],
            )
            spikes = encode(x, cfg["model"]["timesteps"], cfg["model"]["encoding"], generator)
            rates = model(spikes.to(device))
            target = rate_targets(y.to(device), classes)
            total_loss += nn.functional.mse_loss(rates, target).item() * len(y)
            predicted = rates.argmax(1).cpu()
            confusion += torch.bincount(y * classes + predicted, minlength=classes**2).reshape(
                classes, classes
            )
    tp = confusion.diag().double()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator, torch.zeros_like(tp))
    model.train(previous)
    return {
        "accuracy": tp.sum().item() / len(dataset),
        "macro_f1": f1.mean().item(),
        "mse": total_loss / len(dataset),
        "samples": len(dataset),
        "confusion_matrix": confusion.tolist(),
    }


class Client:
    def __init__(self, client_id: int, indices: list[int], global_model: nn.Module, cfg: dict):
        self.client_id = client_id
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.model = deepcopy(global_model)
        self.version = 0
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(cfg["seed"] + 1000 + client_id)
        self.tracker = SaliencyTracker(cfg["aggregation"]["saliency_decay"])
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg["federation"]["learning_rate"]
        )

    def synchronize(self, state: dict, version: int) -> None:
        self.model.load_state_dict(state)
        self.version = version
        self.reset_optimizer()
        self.tracker = SaliencyTracker(self.cfg["aggregation"]["saliency_decay"])

    def train(self, dataset, device: torch.device) -> tuple[float, float]:
        cfg = self.cfg
        before = clone_state(self.model.state_dict())
        x_all, y_all = dataset.tensors
        total_loss, total = 0.0, 0
        self.model.train()
        for _ in range(cfg["federation"]["local_epochs"]):
            order = self.indices[torch.randperm(len(self.indices), generator=self.generator)]
            for batch in order.split(cfg["federation"]["batch_size"]):
                x, y = x_all[batch], y_all[batch].to(device)
                spikes = encode(
                    x, cfg["model"]["timesteps"], cfg["model"]["encoding"], self.generator
                ).to(device)
                self.optimizer.zero_grad(set_to_none=True)
                rates = self.model(spikes)
                loss = nn.functional.mse_loss(rates, rate_targets(y, cfg["data"]["num_classes"]))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite local loss; review neuron discretization")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg["federation"]["clip_grad_norm"],
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                total_loss += loss.item() * len(batch)
                total += len(batch)
        saliency = self.tracker.update(before, clone_state(self.model.state_dict()))
        return total_loss / total, saliency


def run_experiment(cfg: dict, output: str | Path) -> dict:
    validate(cfg)
    cfg = deepcopy(cfg)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new directory.")
    device = seed_runtime(cfg)
    bundle = load_data(cfg)
    f, a, n = (cfg[k] for k in ("federation", "aggregation", "network"))
    shards = dirichlet_partition(
        bundle.train.tensors[1].numpy(),
        f["clients"],
        f["dirichlet_alpha"],
        cfg["seed"] + 202,
        f["min_client_samples"],
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg)
    write_json(output / "data_provenance.json", bundle.provenance)
    write_json(output / "partition.json", {str(i): values for i, values in enumerate(shards)})
    environment = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    write_json(output / "environment.json", environment)
    global_model = build_model(cfg).to(device)
    global_state = clone_state(global_model.state_dict())
    clients = [Client(i, shard, global_model, cfg) for i, shard in enumerate(shards)]
    network = SimulatedNetwork(
        range(f["clients"]),
        seed=cfg["seed"] + 303,
        one_way_ms=n["one_way_ms"],
        jitter_ms=n["jitter_ms"],
        drop_probability=n["drop_probability"],
    )
    protocol = SponsorElection(
        range(f["clients"]), network, n["rpc_timeout_ms"], n["heartbeat_timeout_ms"]
    )
    clock = AggregationClock(a)
    version = uplink = downlink = dense_equivalent = aggregations = 0
    metrics, events = [], []
    started = time.perf_counter()
    initial = evaluate(global_model, bundle.test, cfg, device)

    def record(round_id, leader, online, loss, aggregated, due, forced, score):
        evaluation = evaluate(global_model, bundle.test, cfg, device)
        row = {
            "round": round_id,
            "version": version,
            "leader": leader,
            "online_clients": len(online),
            "aggregation_due": due,
            "aggregated": aggregated,
            "forced_final_flush": forced,
            "train_loss": loss,
            "global_test_accuracy": evaluation["accuracy"],
            "global_test_macro_f1": evaluation["macro_f1"],
            "global_test_mse": evaluation["mse"],
            "saliency": score,
            "next_threshold": clock.threshold,
            "uplink_payload_bytes": uplink,
            "downlink_payload_bytes": downlink,
            "dense_uplink_equivalent_bytes": dense_equivalent,
            "control_bytes_estimated": len(network.messages) * n["control_message_bytes"],
            "control_virtual_ms": network.now_ms,
        }
        metrics.append(row)
        return evaluation

    final = record(0, None, [], None, False, False, False, 0.0)
    for round_id in range(1, f["rounds"] + 1):
        online = {
            i
            for i in range(f["clients"])
            if round_id not in f["offline"].get(str(i), f["offline"].get(i, []))
        }
        network.set_alive(online)
        leader = protocol.ensure_leader()
        loss_sum, trained_samples, score = 0.0, 0, 0.0
        for i in sorted(online):
            client = clients[i]
            if client.version != version:
                # Recovery policy: discard that client's unsynchronized old-version work.
                client.synchronize(global_state, version)
                # A recovering leader also needs the latest logical global model.
                downlink += state_bytes(global_state)
                events.append(
                    {"round": round_id, "event": "client_catchup", "client": i, "version": version}
                )
            loss, client_score = client.train(bundle.train, device)
            score += client_score
            loss_sum += loss * len(client.indices)
            trained_samples += len(client.indices)
        due = clock.observe(score)
        forced = bool(
            not due and f["flush_final"] and round_id == f["rounds"] and clock.elapsed > 0
        )
        aggregated = bool((due or forced) and leader is not None and online)
        if aggregated:
            updates = [
                pack_update(
                    i,
                    len(clients[i].indices),
                    clients[i].version,
                    clone_state(clients[i].model.state_dict()),
                    global_state,
                    clients[i].tracker,
                    a["keep_ratio"],
                )
                for i in sorted(online)
            ]
            uplink += sum(update.nbytes for update in updates if update.client != leader)
            dense_equivalent += state_bytes(global_state) * (len(online) - 1)
            downlink += state_bytes(global_state) * (len(online) - 1)
            threshold_before = clock.threshold
            global_state = aggregate(global_state, updates, version)
            version += 1
            aggregations += 1
            global_model.load_state_dict(global_state)
            for i in sorted(online):
                clients[i].synchronize(global_state, version)
            clock.committed()
            events.append(
                {
                    "round": round_id,
                    "event": "aggregate",
                    "leader": leader,
                    "term": protocol.term,
                    "version": version,
                    "clients": sorted(online),
                    "threshold_before": threshold_before,
                    "forced_final_flush": forced,
                }
            )
        elif due or forced:
            events.append(
                {"round": round_id, "event": "aggregation_deferred", "reason": "no_quorum"}
            )
        final = record(
            round_id,
            leader,
            online,
            loss_sum / trained_samples if trained_samples else None,
            aggregated,
            due,
            forced,
            score,
        )
    summary = {
        "package_version": "0.1.0",
        "data": bundle.provenance,
        "neuron": cfg["model"]["neuron"],
        "aggregation_policy": a["policy"],
        "keep_ratio": a["keep_ratio"],
        "seed": cfg["seed"],
        "rounds": f["rounds"],
        "aggregation_events": aggregations,
        "initial": initial,
        "final": final,
        "uplink_payload_bytes": uplink,
        "downlink_payload_bytes": downlink,
        "dense_uplink_equivalent_bytes": dense_equivalent,
        "control_bytes_estimated": len(network.messages) * n["control_message_bytes"],
        "elapsed_training_wall_seconds": time.perf_counter() - started,
        "uncommitted_local_rounds": clock.elapsed,
        "measurement_scope": "tensor payload only; control size estimated; no physical energy measurement",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "events.json", events)
    write_json(output / "elections.json", protocol.history)
    write_json(output / "network_trace.json", network.trace())
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    torch.save(
        {
            "format_version": 1,
            "model_state": global_state,
            "config": cfg,
            "global_version": version,
            "data_provenance": bundle.provenance,
        },
        output / "model.pt",
    )
    return summary
