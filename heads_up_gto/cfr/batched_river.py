"""Batched DCFR river solver: many range pairs on one board and betting tree at once.

Every spot in a batch shares the board, pot, stacks and action tree. All spots
use the same hand axis (every combo not blocked by the board); a hand outside a
spot's range simply has zero weight, so it never affects anyone else's
strategy. Arrays are ``[spots, hands]`` per decision (regrets: ``[actions,
spots, hands]``), so each numpy call is amortised over the whole batch.

Algorithm and discounts are the same as ``CFRPlusRiver(algorithm='dcfr')``, and
each spot follows the same trajectory as solving it alone, to rounding error
(late iterations can drift where regrets sit on the uniform-strategy threshold). Showdown mass uses the
same O(N) sorted-prefix-sum with card removal.
"""
import math
import time

import numpy as np

from ..cards import COMBOS, score


class _Node:
    __slots__ = ('state', 'actor', 'actions', 'children', 'index')

    def __init__(self, state):
        self.state, self.actor = state, state.actor
        self.actions, self.children, self.index = (), (), -1


class BatchedRiver:
    def __init__(self, root, range_pairs, settings, dtype=np.float64):
        if root.street != 2 or len(root.board) != 5 or root.pending or root.terminal:
            raise ValueError('An active river decision with five board cards is required.')
        if not range_pairs:
            raise ValueError('At least one pair of ranges is required.')
        start = time.perf_counter()
        self.root, self.settings, self.dtype = root, settings, np.dtype(dtype)
        blocked = set(root.board)
        self.hands = [h for h in COMBOS if not blocked.intersection(h)]
        index = {h: i for i, h in enumerate(self.hands)}
        n, b = len(self.hands), len(range_pairs)
        self.size, self.batch = n, b
        weights = np.zeros((2, b, n))
        for k, pair in enumerate(range_pairs):
            if len(pair) != 2:
                raise ValueError('Each spot needs two ranges.')
            for p, r in enumerate(pair):
                for h, w in r.items():
                    if not math.isfinite(w) or w < 0:
                        raise ValueError('Range weights must be nonnegative and finite.')
                    i = index.get(tuple(sorted(h)))
                    if i is not None:
                        weights[p, k, i] = w
        totals = weights.sum(axis=2)
        if (totals <= 0).any():
            raise ValueError('Every spot needs two nonempty, unblocked ranges.')
        self.weights = weights / totals[:, :, None]
        self._walk_weights = self.weights.astype(self.dtype)

        strength = np.asarray([score(h, root.board) for h in self.hands], dtype=object)
        rank = {s: i for i, s in enumerate(sorted(set(strength)))}
        strength = np.asarray([rank[s] for s in strength])
        order = np.argsort(strength, kind='stable')
        ordered = strength[order]
        holders = [[] for _ in range(52)]
        for k, j in enumerate(order):
            for c in self.hands[j]:
                holders[c].append(k)
        holders = [np.asarray(h, dtype=np.int64) for h in holders]
        offset = np.concatenate(([0], np.cumsum([len(h) for h in holders])))
        cards = np.asarray(self.hands)
        lo = np.searchsorted(ordered, strength, 'left')
        hi = np.searchsorted(ordered, strength, 'right')

        def ends(card, bound):
            out = np.empty(len(card), dtype=np.int64)
            for c in np.unique(card):
                m = card == c
                out[m] = offset[c] + np.searchsorted(holders[c], bound[m], 'left')
            return out

        full = np.full(n, n)
        self._order, self._flat = order, order[np.concatenate(holders)]
        self._lo, self._hi = lo, hi
        self._start = [offset[cards[:, 0]], offset[cards[:, 1]]]
        self._end = [[ends(cards[:, s], bound) for bound in (lo, hi, full)] for s in (0, 1)]
        self.normalizer = (self.weights[0] * self._showdown(self.weights[1])[3]).sum(axis=1)
        if (self.normalizer <= 0).any():
            raise ValueError('A spot has ranges with no compatible private deal.')

        self.nodes = []
        self.tree = self._build(root)
        self.regret = [np.zeros((len(x.actions), b, n), self.dtype) if x.actions else None for x in self.nodes]
        self.strategy_sum = [None if r is None else np.zeros_like(r) for r in self.regret]
        self.iterations = 0
        self.stats = {'spots': b, 'hands': n, 'public_decisions': len(self.nodes),
                      'build_seconds': time.perf_counter() - start}

    def _build(self, state):
        node = _Node(state)
        if state.terminal:
            return node
        node.actions = state.actions(self.settings)
        node.index = len(self.nodes)
        self.nodes.append(node)
        node.children = tuple(self._build(state.apply(a)) for a in node.actions)
        return node

    def _showdown(self, reach):
        """Rows win, loss, tie, legal; each ``[spots, hands]``, reach mass per own hand."""
        reach = reach.astype(np.float64, copy=False)
        zero = np.zeros((len(reach), 1))
        prefix = np.concatenate((zero, np.cumsum(reach[:, self._order], axis=1)), axis=1)
        held = np.concatenate((zero, np.cumsum(reach[:, self._flat], axis=1)), axis=1)
        (s1, s2), ((a1, b1, c1), (a2, b2, c2)) = self._start, self._end
        base = held[:, s1] + held[:, s2]
        win = prefix[:, self._lo] - (held[:, a1] + held[:, a2] - base)
        upto = prefix[:, self._hi] - (held[:, b1] + held[:, b2] - base) + reach
        legal = prefix[:, -1:] - (held[:, c1] + held[:, c2] - base) + reach
        return np.stack((win, legal - upto, upto - win, legal)).astype(self.dtype, copy=False)

    def _leaf(self, state, player, reach):
        out = self._showdown(reach)
        if state.terminal == 'fold':
            return state.payoff(state.winner, player) * out[3]
        return (state.payoff(player, player) * out[0] + state.payoff(1 - player, player) * out[1]
                + state.payoff(-1, player) * out[2])

    def _current(self, node):
        positive = np.maximum(self.regret[node.index], 0.)
        total = positive.sum(axis=0)
        return np.where(total > 1e-12, positive / np.maximum(total, 1e-20), 1 / len(node.actions))

    def _average(self, node):
        s = self.strategy_sum[node.index].astype(np.float64)
        total = s.sum(axis=0)
        return np.where(total > 1e-12, s / np.maximum(total, 1e-20), 1 / len(node.actions))

    @staticmethod
    def _discounts(t):
        lower = 0 if t == 0 else 1 << ((t.bit_length() - 1) & ~1)  # 0, 1, 4, 16, ...
        a = max(t - 1, 0) ** 1.5
        g = t - lower
        return a / (a + 1.), .5, (g / (g + 1.)) ** 3

    def _walk(self, node, player, reach, discounts):
        if node.state.terminal:
            return self._leaf(node.state, player, reach)
        i = node.index
        sigma = self._current(node)
        if node.actor == player:
            values = np.empty_like(sigma)
            for a, child in enumerate(node.children):
                values[a] = self._walk(child, player, reach, discounts)
            value = (sigma * values).sum(axis=0)
            alpha, beta, gamma = discounts
            regret = self.regret[i]
            regret *= np.where(regret >= 0, alpha, beta)
            regret += values - value
            self.strategy_sum[i] *= gamma
            self.strategy_sum[i] += sigma
            return value
        return sum(self._walk(child, player, reach * sigma[a], discounts)
                   for a, child in enumerate(node.children))

    def _best_response(self, node, player, reach):
        if node.state.terminal:
            return self._leaf(node.state, player, reach)
        if node.actor == player:
            return np.max([self._best_response(c, player, reach) for c in node.children], axis=0)
        sigma = self._average(node)
        return sum(self._best_response(c, player, reach * sigma[a]) for a, c in enumerate(node.children))

    def _policy_value(self, node, player, reach):
        if node.state.terminal:
            return self._leaf(node.state, player, reach)
        sigma = self._average(node)
        if node.actor == player:
            return sum(sigma[a] * self._policy_value(c, player, reach) for a, c in enumerate(node.children))
        return sum(self._policy_value(c, player, reach * sigma[a]) for a, c in enumerate(node.children))

    def exploitability(self):
        """Per-spot (exploitability, player-0 value, best-response values), all in BB."""
        best = [(self.weights[p] * self._best_response(self.tree, p, self.weights[1 - p])).sum(axis=1)
                / self.normalizer for p in (0, 1)]
        value = (self.weights[0] * self._policy_value(self.tree, 0, self.weights[1])).sum(axis=1) / self.normalizer
        return np.maximum(0., (best[0] + best[1]) / 2), value, best

    def solve(self, *, iterations=1000, target_exploitability=None, check_every=25, time_limit_seconds=None):
        """Run until every spot is at ``target_exploitability`` (BB), the cap, or the time limit."""
        history, spent = [], 0.  # ``spent`` excludes exploitability checks
        for _ in range(iterations):
            began = time.perf_counter()
            discounts = self._discounts(self.iterations)
            self.iterations += 1
            for p in (0, 1):
                self._walk(self.tree, p, self._walk_weights[1 - p], discounts)
            spent += time.perf_counter() - began
            done = time_limit_seconds is not None and spent > time_limit_seconds
            if self.iterations % check_every == 0 or done or self.iterations == iterations:
                exploit = self.exploitability()[0]
                history.append((self.iterations, spent, float(exploit.max())))
                if done or target_exploitability is not None and exploit.max() <= target_exploitability:
                    break
        exploit, value, best = self.exploitability()
        self.stats.update(iterations=self.iterations, exploitability_bb=exploit, value_bb=value,
                          history=history, solve_seconds=spent)
        return self

    def hand_values(self):
        """Per-spot, per-hand on-policy EV (BB) for each player: two ``[spots, hands]`` arrays.

        NaN where a hand has no compatible opponent. Hands outside a spot's
        range still get a (best-response-quality) value; mask by ``weights``.
        """
        out = []
        for p in (0, 1):
            cfv = self._policy_value(self.tree, p, self.weights[1 - p])
            mass = self._showdown(self.weights[1 - p])[3]
            out.append(np.divide(cfv, mass, out=np.full(cfv.shape, np.nan), where=mass > 1e-12))
        return out

    def policy(self, spot, state, hand):
        """Average action probabilities for one hand of one spot at a public state."""
        node = self.tree
        for key in state.path[len(self.root.path):]:
            node = node.children[[a.key for a in node.actions].index(key)]
        return self._average(node)[:, spot, self.hands.index(tuple(sorted(hand)))].tolist()
