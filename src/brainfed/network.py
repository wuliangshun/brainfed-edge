"""Virtual-clock RPC transport with latency, loss, failures and partitions.

Calls in a batch start together. Requests beyond the timeout are discarded;
this bounded simulator does not retain delayed packets across RPC batches.
Data-plane model updates are NOT sent through this transport (see SCOPE.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import random


@dataclass
class Message:
    kind: str
    source: int
    target: int
    sent_ms: float
    delivered: bool
    latency_ms: float


class SimulatedNetwork:
    def __init__(self, members, seed=0, one_way_ms=3.0, jitter_ms=1.0, drop_probability=0.0):
        self.members = tuple(sorted(members))
        self.alive = set(self.members)
        self.blocked: set[tuple[int, int]] = set()
        self.bursts: list[tuple[float, float, float]] = []
        self.rng = random.Random(seed)
        self.one_way_ms = one_way_ms
        self.jitter_ms = jitter_ms
        self.drop_probability = drop_probability
        self.now_ms = 0.0
        self.messages: list[Message] = []

    def set_alive(self, alive) -> None:
        alive = set(alive)
        if not alive <= set(self.members):
            raise ValueError("Unknown network member")
        self.alive = alive

    def partition(self, components: list[set[int]]) -> None:
        all_nodes = set().union(*components)
        if all_nodes != set(self.members) or sum(map(len, components)) != len(all_nodes):
            raise ValueError("Partition components must cover every member exactly once")
        self.blocked = {
            (a, b)
            for i, group in enumerate(components)
            for a in group
            for other in components[i + 1 :]
            for b in other
        }
        self.blocked |= {(b, a) for a, b in list(self.blocked)}

    def _travel(self, source: int, target: int, sent: float) -> tuple[bool, float]:
        delay = self.one_way_ms + self.rng.uniform(0, self.jitter_ms)
        delay += sum(extra for start, end, extra in self.bursts if start <= sent < end)
        ok = (
            source in self.alive
            and target in self.alive
            and (source, target) not in self.blocked
            and self.rng.random() >= self.drop_probability
        )
        return ok, delay

    def rpc_batch(
        self,
        source: int,
        targets,
        kind: str,
        payload: dict,
        handler: Callable[[int, dict], dict],
        timeout_ms: float,
    ) -> dict[int, dict]:
        """Simulate parallel request/reply RPCs; local calls have no wire messages."""
        start = self.now_ms
        responses: dict[int, dict] = {}
        pending = []
        durations = []
        for target in sorted(targets):
            if target == source:
                if source in self.alive:
                    responses[target] = handler(target, payload)
                continue
            ok, forward = self._travel(source, target, start)
            delivered = ok and forward <= timeout_ms
            self.messages.append(Message(kind, source, target, start, delivered, forward))
            if delivered:
                pending.append((start + forward, target))
            else:
                durations.append(timeout_ms)
        for arrived, target in sorted(pending):
            answer = handler(target, payload)
            ok, reverse = self._travel(target, source, arrived)
            elapsed = arrived - start + reverse
            delivered = ok and elapsed <= timeout_ms
            self.messages.append(
                Message(kind + "Reply", target, source, arrived, delivered, reverse)
            )
            if delivered:
                responses[target] = answer
                durations.append(elapsed)
            else:
                durations.append(timeout_ms)
        self.now_ms += max(durations, default=0.0)
        return responses

    def trace(self) -> list[dict]:
        return [asdict(message) for message in self.messages]
