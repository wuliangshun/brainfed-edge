"""Command-line entry points; all demos run without network or data downloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .config import load_config
from .data import load_data
from .election import SponsorElection
from .experiment import evaluate, run_experiment, seed_runtime, write_json
from .model import build_model
from .network import SimulatedNetwork


def election_demo(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, alive, parts, burst in [
        ("normal", {0, 1, 2, 3, 4}, None, False),
        ("node_0_failed", {1, 2, 3, 4}, None, False),
        ("minority_only", {0, 1}, None, False),
        ("partition_minority_sponsor", {0, 1, 2, 3, 4}, [{0, 1}, {2, 3, 4}], False),
        ("burst_delay", {0, 1, 2, 3, 4}, None, True),
    ]:
        network = SimulatedNetwork(range(5), seed=7, one_way_ms=3.0, jitter_ms=1.0)
        network.set_alive(alive)
        if parts:
            network.partition(parts)
        if burst:
            network.bursts = [(0.0, 120.0, 20.0)]
        protocol = SponsorElection(range(5), network)
        result = protocol.elect()
        cases.append({"scenario": name, **result.__dict__})
        write_json(output / f"{name}_trace.json", network.trace())
    write_json(output / "summary.json", {"simulation_only": True, "cases": cases})
    return {"simulation_only": True, "cases": cases}


def compare(config: dict, output: Path, seeds: list[int]) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for neuron in ("lif", "mdn"):
            for policy in ("periodic", "event"):
                cfg = deepcopy(config)
                cfg["seed"], cfg["model"]["neuron"] = seed, neuron
                cfg["aggregation"]["policy"] = policy
                # Controlled pair: sparse event updates vs dense periodic updates.
                cfg["aggregation"]["keep_ratio"] = (
                    1.0 if policy == "periodic" else config["aggregation"]["keep_ratio"]
                )
                name = f"{neuron}_{policy}_seed{seed}"
                summary = run_experiment(cfg, output / name)
                rows.append(
                    {
                        "run": name,
                        "seed": seed,
                        "neuron": neuron,
                        "policy": policy,
                        "accuracy": summary["final"]["accuracy"],
                        "macro_f1": summary["final"]["macro_f1"],
                        "mse": summary["final"]["mse"],
                        "aggregation_events": summary["aggregation_events"],
                        "uplink_payload_bytes": summary["uplink_payload_bytes"],
                        "downlink_payload_bytes": summary["downlink_payload_bytes"],
                    }
                )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for neuron in ("lif", "mdn"):
        for policy in ("periodic", "event"):
            entries = [r for r in rows if r["neuron"] == neuron and r["policy"] == policy]
            grouped.append(
                {
                    "neuron": neuron,
                    "policy": policy,
                    "seeds": seeds,
                    "accuracy_mean": float(np.mean([r["accuracy"] for r in entries])),
                    "accuracy_std_population": float(np.std([r["accuracy"] for r in entries])),
                    "uplink_payload_bytes_mean": float(
                        np.mean([r["uplink_payload_bytes"] for r in entries])
                    ),
                }
            )
    result = {
        "data_source": config["data"]["source"],
        "groups": grouped,
        "comparison_scope": "event timing and sparsification change together; use one-factor configs for isolated ablations",
    }
    write_json(output / "comparison_summary.json", result)
    return result


def evaluate_checkpoint(checkpoint: Path, data_path: str | None, angle: float) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    cfg = saved["config"]
    cfg["device"] = "cpu"
    if data_path:
        cfg["data"]["source"] = "npz"
        cfg["data"]["path"] = str(Path(data_path).expanduser().resolve())
    device = seed_runtime(cfg)
    bundle = load_data(cfg)
    dataset = bundle.test
    if not math.isfinite(angle):
        raise ValueError("Rotation angle must be finite")
    if angle:
        x, y = dataset.tensors
        if cfg["data"]["task"] != "vision" or x.ndim != 2 or x.shape[-1] != 1728:
            raise ValueError("Rotation requires vision data with flattened 3x24x24 images")
        radians = math.radians(angle)
        theta = torch.tensor(
            [[math.cos(radians), -math.sin(radians), 0], [math.sin(radians), math.cos(radians), 0]],
            dtype=x.dtype,
        )
        images = x.reshape(-1, 3, 24, 24)
        grid = torch.nn.functional.affine_grid(
            theta.expand(len(images), -1, -1), images.shape, align_corners=False
        )
        rotated = torch.nn.functional.grid_sample(
            images, grid, align_corners=False, padding_mode="zeros"
        )
        dataset = TensorDataset(rotated.reshape(-1, 1728), y)
    model = build_model(cfg).to(device)
    model.load_state_dict(saved["model_state"])
    return {
        "rotation_degrees": angle,
        "data": bundle.provenance,
        **evaluate(model, dataset, cfg, device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="brainfed")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        if command == "compare":
            p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    p = sub.add_parser("election-demo")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data")
    p.add_argument("--rotation", type=float, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_experiment(load_config(args.config), args.output)
        elif args.command == "compare":
            if len(args.seeds) != len(set(args.seeds)):
                raise ValueError("Comparison seeds must be distinct")
            result = compare(load_config(args.config), args.output, args.seeds)
        elif args.command == "election-demo":
            result = election_demo(args.output)
        else:
            result = evaluate_checkpoint(args.checkpoint, args.data, args.rotation)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"brainfed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
