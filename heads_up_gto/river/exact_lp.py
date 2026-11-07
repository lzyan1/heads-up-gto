"""Experimental river equilibrium oracle using sequence-form linear programming.

No learned values, CFR iterations or perfect-information minimax. Every decision
is shared across indistinguishable opponent hands. Exact means all legal private
pairs and betting branches, with LP solutions certified to numerical tolerance.
The optional ``exact`` extra supplies SciPy/HiGHS. This is not wired into training.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from ..cards import combo_text, remove_blocked, score


@dataclass
class PublicNode:
    state: object
    actions: tuple = ()
    children: tuple = ()
    sequences: object = None
    parent: object = None


class ExactRiver:
    def __init__(self, root, ranges, settings, *, max_payoff_entries=25_000_000):
        from scipy import sparse
        start = time.perf_counter()
        if root.street != 2 or len(root.board) != 5 or root.pending or root.terminal:
            raise ValueError('An active river decision with five board cards is required.')
        self.root, self.settings = root, settings
        if len(ranges) != 2 or any(not math.isfinite(w) or w < 0 for r in ranges for w in r.values()):
            raise ValueError('Two ranges with nonnegative finite weights are required.')
        self.ranges = tuple(remove_blocked(r, root.board) for r in ranges)
        if any(not r for r in self.ranges):
            raise ValueError('Both players need an unblocked range.')
        self.hands = [tuple(r) for r in self.ranges]
        self.hand_indices = [{h: i for i, h in enumerate(hands)} for hands in self.hands]
        sizes = [len(h) for h in self.hands]
        masks = [np.asarray([sum(1 << c for c in h) for h in hands], dtype=np.uint64)
                 for hands in self.hands]
        legal = (masks[0][:, None] & masks[1][None, :]) == 0
        self.pair_i, self.pair_j = np.nonzero(legal)
        if not len(self.pair_i):
            raise ValueError('The ranges have no compatible private deal.')
        weights = [np.asarray(list(r.values()), dtype=np.float64) for r in self.ranges]
        weights = [w / w.sum() for w in weights]
        self.chance = weights[0][self.pair_i] * weights[1][self.pair_j]
        self.chance /= self.chance.sum()
        scores = [[score(h, root.board) for h in hands] for hands in self.hands]
        # Comparing evaluator tuples preserves every kicker and tie exactly.
        ranks = {s: i for i, s in enumerate(sorted(set(scores[0] + scores[1])))}
        strength = [np.asarray([ranks[s] for s in ss]) for ss in scores]
        comparison = np.sign(strength[0][self.pair_i] - strength[1][self.pair_j])
        self.winners = np.where(comparison > 0, 0, np.where(comparison < 0, 1, -1))
        self.sequence_counts = [1, 1]
        self.flow_rows = [[0], [0]]
        self.flow_cols = [[0], [0]]
        self.flow_data = [[1.], [1.]]
        self.flow_counts = [1, 1]
        self.nodes = {}
        leaves = []

        def build(state, previous):
            node = PublicNode(state)
            if state.terminal:
                leaves.append((node, previous))
                if len(leaves) * len(self.chance) > max_payoff_entries:
                    raise ValueError('Payoff-entry guard exceeded; narrow the ranges or increase the explicit guard.')
                return node
            p = state.actor
            node.actions = state.actions(settings)
            node.parent = previous[p]
            count = len(node.actions) * sizes[p]
            node.sequences = np.arange(self.sequence_counts[p], self.sequence_counts[p] + count).reshape(len(node.actions), sizes[p])
            self.sequence_counts[p] += count
            rows = np.arange(self.flow_counts[p], self.flow_counts[p] + sizes[p])
            self.flow_counts[p] += sizes[p]
            self.flow_rows[p].extend(np.tile(rows, len(node.actions) + 1))
            self.flow_cols[p].extend(np.concatenate((previous[p], node.sequences.ravel())))
            self.flow_data[p].extend(np.concatenate((-np.ones(sizes[p]), np.ones(count))))
            self.nodes[state.path] = node
            children = []
            for a, seq in zip(node.actions, node.sequences):
                updated = list(previous)
                updated[p] = seq
                children.append(build(state.apply(a), updated))
            node.children = tuple(children)
            return node

        self.tree = build(root, [np.zeros(n, dtype=int) for n in sizes])
        self.flows = [sparse.coo_matrix((self.flow_data[p], (self.flow_rows[p], self.flow_cols[p])),
                                       shape=(self.flow_counts[p], self.sequence_counts[p])).tocsr()
                      for p in (0, 1)]
        self.rhs = [np.r_[1., np.zeros(n - 1)] for n in self.flow_counts]
        rows, cols, data = [], [], []
        for node, previous in leaves:
            payoff = self.terminal_values(node.state, 0)
            rows.append(previous[0][self.pair_i])
            cols.append(previous[1][self.pair_j])
            data.append(self.chance * payoff)
        self.payoff = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=tuple(self.sequence_counts)).tocsr()
        self.payoff.eliminate_zeros()
        self.policies = {}
        self.stats = {'private_hands': sizes, 'legal_pairs': len(self.chance),
                      'public_decisions': len(self.nodes), 'terminal_histories': len(leaves),
                      'sequences': self.sequence_counts, 'flow_constraints': self.flow_counts,
                      'payoff_nonzeros': self.payoff.nnz,
                      'build_seconds': time.perf_counter() - start}

    def terminal_values(self, state, player):
        if state.terminal == 'fold':
            return np.full(len(self.chance), state.payoff(state.winner, player))
        return np.asarray([state.payoff(w, player) for w in (-1, 0, 1)])[self.winners + 1]

    def solve(self, *, time_limit_seconds=30, tolerance=1e-8):
        from scipy import sparse
        from scipy.optimize import linprog
        start = time.perf_counter()
        a, (e, f), (erhs, frhs) = self.payoff, self.flows, self.rhs
        n0, n1 = self.sequence_counts
        c0, c1 = self.flow_counts
        options = dict(time_limit=time_limit_seconds / 2,
                       primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
        # max f'v subject to Ex=e, x>=0, A'x >= F'v.
        p0 = linprog(np.r_[np.zeros(n0), -frhs],
                     A_ub=sparse.hstack((-a.T, f.T), format='csr'), b_ub=np.zeros(n1),
                     A_eq=sparse.hstack((e, sparse.csr_matrix((c0, c1))), format='csr'), b_eq=erhs,
                     bounds=[(0, None)] * n0 + [(None, None)] * c1, method='highs', options=options)
        if not p0.success:
            raise RuntimeError(f'Player 0 LP: {p0.message}')
        # min e'u subject to Fy=f, y>=0, Ay <= E'u.
        p1 = linprog(np.r_[np.zeros(n1), erhs],
                     A_ub=sparse.hstack((a, -e.T), format='csr'), b_ub=np.zeros(n0),
                     A_eq=sparse.hstack((f, sparse.csr_matrix((c1, c0))), format='csr'), b_eq=frhs,
                     bounds=[(0, None)] * n1 + [(None, None)] * c0, method='highs', options=options)
        if not p1.success:
            raise RuntimeError(f'Player 1 LP: {p1.message}')
        plans = [np.maximum(p0.x[:n0], 0), np.maximum(p1.x[:n1], 0)]
        for path, node in self.nodes.items():
            plan = plans[node.state.actor]
            values = plan[node.sequences].T
            total = values.sum(axis=1, keepdims=True)
            probs = np.divide(values, total,
                              out=np.full(values.shape, 1 / len(node.actions)), where=total > 1e-12)
            self.policies[path] = probs
        self.stats.update(lp_seconds=time.perf_counter() - start,
                          value_bb=float(plans[0] @ (a @ plans[1])),
                          lp_lower_bound_bb=float(-p0.fun), lp_upper_bound_bb=float(p1.fun),
                          lp_gap_bb=float(p1.fun + p0.fun),
                          flow_residual=max(float(np.max(np.abs(flow @ plan - rhs)))
                                            for flow, plan, rhs in zip(self.flows, plans, self.rhs)))
        check = time.perf_counter()
        values, br = self.certify()
        self.stats.update(br, verification_seconds=time.perf_counter() - check)
        self.stats['total_seconds'] = sum(self.stats[k] for k in ('build_seconds', 'lp_seconds', 'verification_seconds'))
        errors = [abs(self.stats['lp_gap_bb']), self.stats['flow_residual'],
                  abs(self.stats['value_bb'] - self.stats['recursive_value_bb']),
                  *(abs(gap) for gap in self.stats['deviation_gains_bb'])]
        if not all(math.isfinite(error) and error <= tolerance for error in errors):
            raise RuntimeError(f'Numerical certificate exceeds {tolerance}: {self.stats}')
        return self

    def policy(self, state, hand):
        index = self.hand_indices[state.actor][tuple(sorted(hand))]
        return self.policies[state.path][index].tolist(), 0, 0

    def certify(self):
        """Independent recursive best responses, grouping by own hand at each node.

        No optimizing action may depend on the opponent's hidden hand. Returns
        on-policy per-deal values as well as both players' best-response bounds.
        """
        pair_indices = (self.pair_i, self.pair_j)
        def walk(node, reach, player, best_response):
            state = node.state
            if state.terminal:
                return self.terminal_values(state, player)
            probabilities = self.policies[state.path][pair_indices[state.actor]].T
            if best_response and state.actor == player:
                children = np.asarray([walk(c, reach, player, True) for c in node.children])
                scores = np.asarray([np.bincount(pair_indices[player], weights=reach * v,
                                                 minlength=len(self.hands[player])) for v in children])
                chosen = np.argmax(scores, axis=0)
                return children[chosen[pair_indices[player]], np.arange(len(self.chance))]
            return sum(prob * walk(c, reach * prob, player, best_response)
                       for prob, c in zip(probabilities, node.children))
        values = walk(self.tree, self.chance, 0, False)
        best = [float(self.chance @ walk(self.tree, self.chance, p, True)) for p in (0, 1)]
        ev = float(self.chance @ values)
        gaps = [best[0] - ev, best[1] + ev]
        return values, {'recursive_value_bb': ev, 'best_response_values_bb': best,
                        'deviation_gains_bb': gaps, 'exploitability_bb': max(0., sum(best) / 2)}

    def root_policy(self):
        return [{'hand': combo_text(h), 'probabilities': self.policies[self.root.path][i].tolist()}
                for i, h in enumerate(self.hands[self.root.actor])]

    def value_for_hand(self, player, hand):
        """Best-response continuation value for one private hand, including zero-reach hands.

        The opponent keeps the solved equilibrium policy; their private hand is
        integrated out with blockers. The maximizing action is shared across
        all opponent cards, never selected with knowledge of those hidden cards.
        """
        opponent = 1 - player
        hands = self.hands[opponent]
        legal = np.asarray([not set(hand).intersection(h) for h in hands])
        weights = np.asarray([self.ranges[opponent][h] for h in hands]) * legal
        if not weights.sum():
            raise ValueError('This private hand has no compatible opponent at the river boundary.')
        weights /= weights.sum()
        hero = score(hand, self.root.board)
        winners = np.asarray([player if hero > score(h, self.root.board) else opponent
                              if hero < score(h, self.root.board) else -1 for h in hands])
        def walk(node, reach):
            state = node.state
            if state.terminal:
                if state.terminal == 'fold':
                    return np.full(len(hands), state.payoff(state.winner, player))
                return np.asarray([state.payoff(w, player) for w in (-1, 0, 1)])[winners + 1]
            if state.actor == player:
                children = [walk(child, reach) for child in node.children]
                return children[int(np.argmax([reach @ child for child in children]))]
            probabilities = self.policies[state.path].T
            return sum(p * walk(child, reach * p) for p, child in zip(probabilities, node.children))
        return float(weights @ walk(self.tree, weights))

    def compact(self):
        """Discard LP/certificate work arrays before caching or crossing a process pipe."""
        for name in ('payoff', 'flows', 'flow_rows', 'flow_cols', 'flow_data',
                     'pair_i', 'pair_j', 'chance', 'winners'):
            self.__dict__.pop(name, None)
        return self
