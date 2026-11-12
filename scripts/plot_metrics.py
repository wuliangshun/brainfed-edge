"""Plot actual run logs; never generate illustrative benchmark measurements."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for run in args.runs:
        with (run / "metrics.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rounds = [int(r["round"]) for r in rows]
        axes[0].plot(rounds, [float(r["global_test_accuracy"]) for r in rows], label=run.name)
        axes[1].step(
            rounds,
            [
                (int(r["uplink_payload_bytes"]) + int(r["downlink_payload_bytes"])) / 1024
                for r in rows
            ],
            where="post",
            label=run.name,
        )
    axes[0].set(xlabel="Local training round", ylabel="Global test accuracy", ylim=(0, 1))
    axes[1].set(xlabel="Local training round", ylabel="Cumulative tensor payload (KiB)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
