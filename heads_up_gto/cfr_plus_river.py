"""Vectorized full-width CFR+ river solver.

The board is fixed on the river, so the only private information is each
player's hand. Every iteration walks the public betting tree once and updates
all hands at once: regrets and strategies are ``[hands, actions]`` arrays, reach
probabilities and counterfactual values are length-``hands`` vectors. There is
no chance sampling. Showdown/fold leaves use one stacked matrix product with
card removal (blockers) baked into the matrices.

Updates alternate between players (CFR+): regrets are floored at zero and the
average strategy is weighted linearly by iteration.
"""
import math
import time

import numpy as np

from .cards import combo_text, remove_blocked, score


class _Node:
    __slots__ = ('state', 'actor', 'actions', 'children', 'index')

    def __init__(self, state):
        self.state, self.actor = state, state.actor
        self.actions, self.children, self.index = (), (), -1


class CFRPlusRiver:
    def __init__(self, root, ranges, settings):
        if root.street != 2 or len(root.board) != 5 or root.pending or root.terminal:
            raise ValueError('An active river decision with five board cards is required.')
        if len(ranges) != 2 or any(not math.isfinite(w) or w < 0 for r in ranges for w in r.values()):
            raise ValueError('Two ranges with nonnegative finite weights are required.')
        start = time.perf_counter()
        self.root, self.settings = root, settings
        self.ranges = tuple(remove_blocked(r, root.board) for r in ranges)
        if any(not r for r in self.ranges):
            raise ValueError('Both players need an unblocked range.')
        self.hands = [tuple(r) for r in self.ranges]
        self.hand_indices = [{h: i for i, h in enumerate(hs)} for hs in self.hands]
        self.sizes = [len(h) for h in self.hands]
        weights = [np.asarray(list(r.values()), dtype=np.float64) for r in self.ranges]
        self.weights = [w / w.sum() for w in weights]
        masks = [np.asarray([sum(1 << c for c in h) for h in hs], dtype=np.uint64) for hs in self.hands]
        legal = (masks[0][:, None] & masks[1][None, :]) == 0
        if not legal.any():
            raise ValueError('The ranges have no compatible private deal.')
        scores = [[score(h, root.board) for h in hs] for hs in self.hands]
        rank = {s: i for i, s in enumerate(sorted(set(scores[0] + scores[1])))}
        s0, s1 = (np.asarray([rank[s] for s in ss]) for ss in scores)
        win = legal & (s0[:, None] > s1[None, :])
        loss = legal & (s0[:, None] < s1[None, :])
        tie = legal & (s0[:, None] == s1[None, :])
        f = np.float64
        # Row blocks: [win, loss, tie, legal] from the owning player's view.
        self.stack = [np.vstack([m.astype(f) for m in (win, loss, tie, legal)]),
                      np.vstack([m.T.astype(f) for m in (loss, win, tie, legal)])]
        self.normalizer = float(self.weights[0] @ (legal.astype(f) @ self.weights[1]))
        self.nodes = []
        self.tree = self._build(root)
        self.regret = [np.zeros((len(n.actions), self.sizes[n.actor])) if n.actions else None
                       for n in self.nodes]
        self.strategy_sum = [None if r is None else np.zeros_like(r) for r in self.regret]
        self.iterations = 0
        self.stats = {'private_hands': self.sizes, 'public_decisions': len(self.nodes),
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

    # -- strategy -----------------------------------------------------------
    def _current(self, node):
        positive = self.regret[node.index]  # CFR+ keeps regrets nonnegative
        total = positive.sum(axis=0)
        return np.where(total > 1e-12, positive / np.maximum(total, 1e-300), 1 / len(node.actions))

    def _average(self, node):
        s = self.strategy_sum[node.index]
        total = s.sum(axis=0)
        return np.where(total > 1e-12, s / np.maximum(total, 1e-300), 1 / len(node.actions))

    def _leaf(self, state, player, opp_reach):
        out = (self.stack[player] @ opp_reach).reshape(4, -1)
        if state.terminal == 'fold':
            return state.payoff(state.winner, player) * out[3]
        return (state.payoff(player, player) * out[0] + state.payoff(1 - player, player) * out[1]
                + state.payoff(-1, player) * out[2])

    # -- one CFR+ traversal for ``player`` ------------------------------------
    def _walk(self, node, player, reach, opp_reach, t):
        if node.state.terminal:
            return self._leaf(node.state, player, opp_reach)
        i = node.index
        if node.actor == player:
            sigma = self._current(node)
            values = np.empty_like(sigma)
            for a, child in enumerate(node.children):
                values[a] = self._walk(child, player, reach * sigma[a], opp_reach, t)
            value = (sigma * values).sum(axis=0)
            self.regret[i] = np.maximum(self.regret[i] + values - value, 0.)
            self.strategy_sum[i] += t * reach * sigma
            return value
        sigma = self._current(node)
        return sum(self._walk(child, player, reach, opp_reach * sigma[a], t)
                   for a, child in enumerate(node.children))

    def _best_response(self, node, player, opp_reach):
        if node.state.terminal:
            return self._leaf(node.state, player, opp_reach)
        if node.actor == player:
            return np.max([self._best_response(c, player, opp_reach) for c in node.children], axis=0)
        sigma = self._average(node)
        return sum(self._best_response(c, player, opp_reach * sigma[a]) for a, c in enumerate(node.children))

    def _policy_value(self, node, player, reach, opp_reach):
        if node.state.terminal:
            return self._leaf(node.state, player, opp_reach)
        sigma = self._average(node)
        if node.actor == player:
            return sum(sigma[a] * self._policy_value(c, player, reach * sigma[a], opp_reach)
                       for a, c in enumerate(node.children))
        return sum(self._policy_value(c, player, reach, opp_reach * sigma[a])
                   for a, c in enumerate(node.children))

    def exploitability(self):
        """Mean best-response gain in BB over the current average strategy."""
        best = [float(self.weights[p] @ self._best_response(self.tree, p, self.weights[1 - p]))
                / self.normalizer for p in (0, 1)]
        ev = float(self.weights[0] @ self._policy_value(self.tree, 0, self.weights[0], self.weights[1])) / self.normalizer
        return max(0., (best[0] + best[1]) / 2), ev, best

    def solve(self, *, iterations=1000, target_exploitability=None, check_every=50, time_limit_seconds=None):
        """Run CFR+ until ``iterations``, the exploitability target (BB) or the time limit."""
        start = time.perf_counter()
        history = []
        for _ in range(iterations):
            self.iterations += 1
            for p in (0, 1):
                self._walk(self.tree, p, self.weights[p], self.weights[1 - p], self.iterations)
            done = time_limit_seconds is not None and time.perf_counter() - start > time_limit_seconds
            if self.iterations % check_every == 0 or done or self.iterations == iterations:
                exploit, ev, _ = self.exploitability()
                history.append((self.iterations, exploit))
                if done or target_exploitability is not None and exploit <= target_exploitability:
                    break
        exploit, ev, best = self.exploitability()
        self.stats.update(iterations=self.iterations, exploitability_bb=exploit, value_bb=ev,
                          best_response_values_bb=best, history=history,
                          solve_seconds=time.perf_counter() - start)
        return self

    # -- same read-only API as ExactRiver -----------------------------------
    def policy(self, state, hand):
        node = self._locate(state.path)
        index = self.hand_indices[state.actor][tuple(sorted(hand))]
        return self._average(node)[:, index].tolist(), 0, 0

    def _locate(self, path):
        node = self.tree
        for key in path[len(self.root.path):]:
            node = node.children[[a.key for a in node.actions].index(key)]
        return node

    def root_policy(self):
        avg = self._average(self.tree)
        return [{'hand': combo_text(h), 'probabilities': avg[:, i].tolist()}
                for i, h in enumerate(self.hands[self.root.actor])]
