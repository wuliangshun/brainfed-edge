"""Feature-to-spike reference encoders; raw sensor preprocessing is external."""

import torch
from torch import Tensor


def encode(features: Tensor, timesteps: int, method: str, generator=None) -> Tensor:
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    if features.ndim == 3:
        if method != "preencoded" or features.shape[1] != timesteps:
            raise ValueError("Temporal NPZ data requires preencoded and matching timesteps")
        return features.transpose(0, 1).contiguous()
    if features.ndim != 2:
        raise ValueError("Expected features [batch, features]")
    x = features.clamp(0, 1)
    if method == "rate":
        noise = torch.rand((timesteps, *x.shape), generator=generator, device=x.device)
        return (noise < x.unsqueeze(0)).to(x.dtype)
    time = torch.arange(timesteps, device=x.device)[:, None, None]
    if method == "latency":
        first = torch.round((1 - x) * (timesteps - 1)).long()
        return ((time == first) & (x > 0)).to(x.dtype)
    if method == "phase":
        period = min(8, timesteps)
        phase = torch.round((1 - x) * (period - 1)).long()
        return (((time % period) == phase) & (x > 0)).to(x.dtype)
    if method == "burst":
        burst_limit = min(4, timesteps)
        count = torch.ceil(x * burst_limit).long()
        first = torch.round((1 - x) * (timesteps - burst_limit)).long()
        return ((time >= first) & (time < first + count)).to(x.dtype)
    raise ValueError(f"Unsupported encoding: {method}")
