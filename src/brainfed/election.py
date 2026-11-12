"""Sponsor-based, fixed-membership leader-election reference simulator.

This implements election and majority fencing only, not Raft log replication.
The simulation driver serializes election attempts and chooses one Sponsor.
It cannot establish safety or liveness for arbitrary concurrent real networks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .network import SimulatedNetwork


@dataclass
class Node:
    node_id: int
    term: int = 0
    voted_for: int | None = None
    leader: int | None = None

    def vote(self, term: int, candidate: int) -> bool:
        if term < self.term:
            return False
        if term > self.term:
            self.term, self.voted_for, self.leader = term, None, None
        if self.voted_for not in (None, candidate):
            return False
        self.voted_for = candidate
        return True


@dataclass
class ElectionResult:
    leader: int | None
    term: int
    sponsor: int | None
    elapsed_ms: float
    wire_messages: int
    reason: str


class SponsorElection:
    def __init__(
        self, members, network: SimulatedNetwork, rpc_timeout_ms=80.0, heartbeat_timeout_ms=30.0
    ):
        self.members = tuple(sorted(members))
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("Membership must be nonempty and unique")
        if set(self.members) != set(network.members):
            raise ValueError("Transport and election membership differ")
        self.nodes = {node: Node(node) for node in self.members}
        self.network = network
        self.quorum = len(self.members) // 2 + 1
        self.rpc_timeout_ms = rpc_timeout_ms
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self.leader: int | None = None
        self.term = 0
        self.history: list[dict] = []

    def _change(self, node: int, payload: dict) -> dict:
        peer = self.nodes[node]
        ok = peer.vote(payload["term"], payload["candidate"])
        return {"ok": ok, "term": peer.term}

    def _confirm(self, node: int, payload: dict) -> dict:
        candidate = self.nodes[node]
        voters = set(payload["voters"])
        ok = (
            node == payload["candidate"]
            and candidate.term == payload["term"]
            and candidate.voted_for == node
            and voters <= set(self.members)
            and len(voters) >= self.quorum
        )
        # Voter IDs represent trusted replies in this simulator, not signatures.
        return {"ok": ok, "term": candidate.term}

    def _commit(self, node: int, payload: dict) -> dict:
        peer = self.nodes[node]
        ok = peer.term < payload["term"] or (
            peer.term == payload["term"] and peer.voted_for in (None, payload["candidate"])
        )
        if ok:
            peer.term = payload["term"]
            peer.voted_for = payload["candidate"]
            peer.leader = payload["candidate"]
        return {"ok": ok, "term": peer.term}

    def elect(self, sponsor: int | None = None) -> ElectionResult:
        start, count = self.network.now_ms, len(self.network.messages)
        if sponsor is None:
            # Driver arbitration uses visible alive IDs; no distributed minimum-ID service.
            sponsor = min(self.network.alive, default=None)
        if sponsor is not None and sponsor not in self.network.alive:
            raise ValueError("Sponsor must be alive")
        self.leader = None
        current = self.nodes[sponsor].term if sponsor is not None else self.term
        reason = "no_alive_nodes" if sponsor is None else "no_quorum_or_confirm"
        if sponsor is not None:
            for _ in range(2 * len(self.members) + 2):
                current += 1
                candidate = self.members[(current - 1) % len(self.members)]
                payload = {"term": current, "candidate": candidate}
                replies = self.network.rpc_batch(
                    sponsor,
                    self.members,
                    "ChangeLeaderRPC",
                    payload,
                    self._change,
                    self.rpc_timeout_ms,
                )
                maximum = max((r["term"] for r in replies.values()), default=current)
                voters = [n for n, r in replies.items() if r["ok"] and r["term"] == current]
                if maximum > current:
                    current = maximum
                    continue
                if len(voters) < self.quorum:
                    reason = "no_majority"
                    break
                confirm = self.network.rpc_batch(
                    sponsor,
                    [candidate],
                    "ConfirmRPC",
                    {**payload, "voters": voters},
                    self._confirm,
                    self.rpc_timeout_ms,
                )
                if not confirm.get(candidate, {}).get("ok"):
                    current = max(current, confirm.get(candidate, {}).get("term", current))
                    continue
                commits = self.network.rpc_batch(
                    sponsor,
                    self.members,
                    "LeaderCommit",
                    payload,
                    self._commit,
                    self.rpc_timeout_ms,
                )
                if sum(r["ok"] for r in commits.values()) >= self.quorum:
                    self.leader, self.term, reason = candidate, current, "elected"
                    break
                reason = "commit_not_observed_by_majority"
        self.term = max(self.term, current)
        result = ElectionResult(
            self.leader,
            self.term,
            sponsor,
            self.network.now_ms - start,
            len(self.network.messages) - count,
            reason,
        )
        self.history.append(asdict(result))
        return result

    def heartbeat(self) -> bool:
        if self.leader is None or self.leader not in self.network.alive:
            return False
        payload = {"term": self.term, "candidate": self.leader}
        replies = self.network.rpc_batch(
            self.leader, self.members, "Heartbeat", payload, self._commit, self.heartbeat_timeout_ms
        )
        return sum(r["ok"] for r in replies.values()) >= self.quorum

    def ensure_leader(self) -> int | None:
        if not self.heartbeat():
            self.elect()
        return self.leader
