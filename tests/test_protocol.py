import unittest

from brainfed.election import Node, SponsorElection
from brainfed.network import SimulatedNetwork


class ElectionTests(unittest.TestCase):
    def protocol(self, seed=4):
        network = SimulatedNetwork(range(5), seed=seed)
        return SponsorElection(range(5), network), network

    def test_normal_election_and_heartbeat(self):
        protocol, _ = self.protocol()
        result = protocol.elect()
        self.assertEqual((result.leader, result.term, result.sponsor), (0, 1, 0))
        self.assertTrue(protocol.heartbeat())

    def test_failed_first_candidate_is_skipped_with_new_term(self):
        protocol, network = self.protocol()
        network.set_alive({1, 2, 3, 4})
        result = protocol.elect()
        self.assertEqual(result.leader, 1)
        self.assertEqual(result.term, 2)
        self.assertEqual(protocol.nodes[0].term, 0)

    def test_minority_cannot_elect_even_if_every_alive_node_agrees(self):
        protocol, network = self.protocol()
        network.set_alive({0, 1})
        self.assertIsNone(protocol.elect().leader)

    def test_empty_network_cannot_elect(self):
        protocol, network = self.protocol()
        network.set_alive(set())
        self.assertIsNone(protocol.ensure_leader())

    def test_partition_fences_old_leader_and_majority_can_elect(self):
        protocol, network = self.protocol()
        self.assertEqual(protocol.elect().leader, 0)
        network.partition([{0, 1}, {2, 3, 4}])
        self.assertFalse(protocol.heartbeat())
        self.assertIsNone(protocol.elect(sponsor=0).leader)
        result = protocol.elect(sponsor=2)
        self.assertEqual(result.leader, 2)
        self.assertTrue(protocol.heartbeat())

    def test_one_vote_per_term_and_no_term_rollback(self):
        node = Node(0)
        self.assertTrue(node.vote(2, 1))
        self.assertTrue(node.vote(2, 1))
        self.assertFalse(node.vote(2, 3))
        self.assertFalse(node.vote(1, 1))
        self.assertTrue(node.vote(3, 3))

    def test_commit_does_not_overwrite_conflicting_same_term_vote(self):
        protocol, _ = self.protocol()
        protocol.nodes[1].vote(2, 3)
        response = protocol._commit(1, {"term": 2, "candidate": 0})
        self.assertFalse(response["ok"])
        self.assertEqual(protocol.nodes[1].voted_for, 3)

    def test_higher_peer_term_moves_sponsor_forward(self):
        protocol, _ = self.protocol()
        protocol.nodes[3].term = 20
        result = protocol.elect()
        self.assertGreater(result.term, 20)
        self.assertIsNotNone(result.leader)

    def test_burst_delay_increases_virtual_time(self):
        normal, _ = self.protocol()
        delayed, network = self.protocol()
        network.bursts = [(0, 1000, 10)]
        self.assertGreater(delayed.elect().elapsed_ms, normal.elect().elapsed_ms)

    def test_seeded_trace_is_repeatable(self):
        a, na = self.protocol()
        b, nb = self.protocol()
        self.assertEqual(a.elect(), b.elect())
        self.assertEqual(na.trace(), nb.trace())


if __name__ == "__main__":
    unittest.main()
