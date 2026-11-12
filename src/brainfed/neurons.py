"""LIF and second-order meta-dynamic neurons with surrogate gradients."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BoxSpike(torch.autograd.Function):
    """Hard threshold forward; rectangular surrogate derivative backward."""

    @staticmethod
    def forward(ctx, value: Tensor, width: float) -> Tensor:
        ctx.save_for_backward(value)
        ctx.width = width
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (value,) = ctx.saved_tensors
        return grad_output * (value.abs() < ctx.width).to(value.dtype), None


class SpikingCell(nn.Module):
    """Sequence state is owned by the caller, never shared across minibatches.

    MDN Euler update:
        dU/dt = (U**2 - U - H + I) / tau
        dH/dt = eta_a * (eta_b * U - H)
    After a spike U=eta_c and H+=eta_d. State clipping is an explicit
    numerical guard for this reference discretization, not an additional
    theoretical claim about the continuous dynamics.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = dict(config)

    def forward(self, current: Tensor, state: tuple[Tensor, Tensor] | None = None):
        c = self.cfg
        voltage, recovery = (
            state if state is not None else (torch.zeros_like(current), torch.zeros_like(current))
        )
        drive = c["current_gain"] * current
        if c["neuron"] == "lif":
            next_voltage = c["lif_decay"] * voltage + (1 - c["lif_decay"]) * drive
            next_recovery = recovery
        else:
            safe_voltage = voltage.clamp(-2.0, 2.0)
            next_voltage = voltage + c["dt"] / c["tau"] * (
                safe_voltage.square() - safe_voltage - recovery + drive
            )
            next_recovery = recovery + c["dt"] * c["eta_a"] * (c["eta_b"] * voltage - recovery)
        next_voltage = next_voltage.clamp(-3.0, 3.0)
        spikes = BoxSpike.apply(next_voltage - c["threshold"], c["surrogate_width"])
        reset = spikes.detach()
        next_voltage = next_voltage * (1 - reset) + c["eta_c"] * reset
        if c["neuron"] == "mdn":
            next_recovery = (next_recovery + c["eta_d"] * reset).clamp(-3.0, 3.0)
        return spikes, (next_voltage, next_recovery)
