"""Two-layer SNN. BPTT spans all time steps within each forward call."""

import torch
from torch import Tensor, nn

from .neurons import SpikingCell


class SpikingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, config: dict):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        nn.init.constant_(self.fc1.bias, 0.2)
        nn.init.constant_(self.fc2.bias, 0.2)
        self.hidden_cell = SpikingCell(config)
        self.output_cell = SpikingCell(config)

    def forward(self, spikes: Tensor) -> Tensor:
        if spikes.ndim != 3 or spikes.shape[-1] != self.fc1.in_features:
            raise ValueError("Expected [time, batch, input_features] spike tensor")
        hidden_state = output_state = None
        outputs = []
        for inputs in spikes.unbind(0):
            hidden, hidden_state = self.hidden_cell(self.fc1(inputs), hidden_state)
            output, output_state = self.output_cell(self.fc2(hidden), output_state)
            outputs.append(output)
        return torch.stack(outputs).mean(0)


def build_model(cfg: dict) -> SpikingMLP:
    return SpikingMLP(
        cfg["data"]["input_dim"],
        cfg["model"]["hidden_dim"],
        cfg["data"]["num_classes"],
        cfg["model"],
    )


def rate_targets(labels: Tensor, classes: int) -> Tensor:
    return torch.nn.functional.one_hot(labels, classes).float() * 0.75 + 0.05
