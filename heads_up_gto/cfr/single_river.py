"""Vectorized full-width river solver (DCFR by default, CFR+ optional).

The board is fixed on the river, so the only private information is each
player's hand. Every iteration walks the public betting tree once and updates
all hands at once: regrets and strategies are ``[hands, actions]`` arrays, reach
probabilities and counterfactual values are length-``hands`` vectors. There is
no chance sampling.

Leaves cost O(N), not O(N^2): the opponent's reach is sorted by hand strength
once, and each hand's win/loss/tie/legal mass is a prefix sum with card removal
by inclusion-exclusion over the two cards it holds (plus the identical combo,
which shares both). Updates alternate between players.

``algorithm='dcfr'`` follows b-inary/postflop-solver: regrets are discounted by
t^1.5/(t^1.5+1) when positive and by 0.5 when negative, and the cumulative
strategy is multiplied by (t'/(t'+1))^3 each step, restarting at powers of four.
``'cfr+'`` floors regrets at zero and averages with linear weights.
"""
import math
import time

import numpy as np

from ..cards import combo_text, remove_blocked, score


class _Node:
    __slots__ = ('state', 'actor', 'actions', 'children', 'index')

    def __init__(self, state):
        self.state, self.actor = state, state.actor
        self.actions, self.children, self.index = (), (), -1


class CFRPlusRiver:
    def __init__(self, root, ranges, settings, algorithm='dcfr', dtype=np.float64):
        if algorithm not in ('dcfr', 'cfr+'):
            raise ValueError("algorithm must be 'dcfr' or 'cfr+'.")
        self.algorithm, self.dtype = algorithm, np.dtype(dtype)
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
        scores = [[score(h, root.board) for h in hs] for hs in self.hands]
        rank = {sc: i for i, sc in enumerate(sorted(set(scores[0] + scores[1])))}
        strength = [np.asarray([rank[sc] for sc in ss]) for ss in scores]
        self._prep = []
        for p in (0, 1):
            o = 1 - p
            order = np.argsort(strength[o], kind='stable')
            ordered = strength[o][order]
            holders = [[] for _ in range(52)]  # sorted positions of opponent hands holding each card
            for k, j in enumerate(order):
                for c in self.hands[o][j]:
                    holders[c].append(k)
            holders = [np.asarray(h, dtype=np.int64) for h in holders]
            offset = np.concatenate(([0], np.cumsum([len(h) for h in holders])))
            flat = order[np.concatenate(holders)]
            same = np.asarray([self.hand_indices[o].get(tuple(sorted(h)), -1) for h in self.hands[p]])
            cards = np.asarray([sorted(h) for h in self.hands[p]])
            lo = np.searchsorted(ordered, strength[p], 'left')
            hi = np.searchsorted(ordered, strength[p], 'right')

            def ends(card, bound):
                # index into the leading-zero cumsum of reach over ``flat`` after the
                # holders of ``card`` that sort before ``bound``
                out = np.empty(len(card), dtype=np.int64)
                for c in np.unique(card):
                    m = card == c
                    out[m] = offset[c] + np.searchsorted(holders[c], bound[m], 'left')
                return out

            full = len(order)
            self._prep.append(dict(
                order=order, flat=flat, same=same, lo=lo, hi=hi,
                start=[offset[cards[:, 0]], offset[cards[:, 1]]],
                end=[[ends(cards[:, s], np.full(len(lo), b) if isinstance(b, int) else b)
                      for b in (lo, hi, full)] for s in (0, 1)]))
        self.normalizer = float(self.weights[0] @ self._showdown(0, self.weights[1])[3])
        if self.normalizer <= 0:
            raise ValueError('The ranges have no compatible private deal.')
        self.nodes = []
        self.tree = self._build(root)
        self._walk_weights = [w.astype(self.dtype) for w in self.weights]
        self.regret = [np.zeros((len(n.actions), self.sizes[n.actor]), self.dtype) if n.actions else None
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
        positive = np.maximum(self.regret[node.index], 0.)
        total = positive.sum(axis=0)
        return np.where(total > 1e-12, positive / np.maximum(total, 1e-20), 1 / len(node.actions))

    def _average(self, node):
        s = self.strategy_sum[node.index].astype(np.float64)
        total = s.sum(axis=0)
        return np.where(total > 1e-12, s / np.maximum(total, 1e-300), 1 / len(node.actions))

    def _showdown(self, player, opp_reach):
        """Rows win, loss, tie, legal: per own hand, opponent reach mass in each class."""
        q = self._prep[player]
        opp_reach = opp_reach.astype(np.float64, copy=False)  # cumsums/subtractions stay f64
        prefix = np.concatenate(([0.], np.cumsum(opp_reach[q['order']])))
        held = np.concatenate(([0.], np.cumsum(opp_reach[q['flat']])))
        (s1, s2), ((a1, b1, c1), (a2, b2, c2)) = q['start'], q['end']
        same = np.where(q['same'] >= 0, opp_reach[np.maximum(q['same'], 0)], 0.)
        base = held[s1] + held[s2]
        win = prefix[q['lo']] - (held[a1] + held[a2] - base)
        upto = prefix[q['hi']] - (held[b1] + held[b2] - base) + same
        legal = prefix[-1] - (held[c1] + held[c2] - base) + same
        return np.stack((win, legal - upto, upto - win, legal)).astype(self.dtype, copy=False)

    def _leaf(self, state, player, opp_reach):
        out = self._showdown(player, opp_reach)
        if state.terminal == 'fold':
            return state.payoff(state.winner, player) * out[3]
        return (state.payoff(player, player) * out[0] + state.payoff(1 - player, player) * out[1]
                + state.payoff(-1, player) * out[2])

    # -- one CFR+ traversal for ``player`` ------------------------------------
    def _discounts(self, t):
        """(alpha, beta, gamma) for 0-based iteration ``t``."""
        if self.algorithm == 'cfr+':
            return 1., 0., (t + 1.) / (t + 2.)  # strategy sum gets weight ~t through the recurrence
        lower = 0 if t == 0 else 1 << ((t.bit_length() - 1) & ~1)  # 0, 1, 4, 16, ...
        a = max(t - 1, 0) ** 1.5
        g = t - lower
        return a / (a + 1.), .5, (g / (g + 1.)) ** 3

    def _walk(self, node, player, opp_reach, discounts):
        if node.state.terminal:
            return self._leaf(node.state, player, opp_reach)
        i = node.index
        sigma = self._current(node)
        if node.actor == player:
            values = np.empty_like(sigma)
            for a, child in enumerate(node.children):
                values[a] = self._walk(child, player, opp_reach, discounts)
            value = (sigma * values).sum(axis=0)
            alpha, beta, gamma = discounts
            regret = self.regret[i]
            regret *= np.where(regret >= 0, alpha, beta)
            regret += values - value
            if self.algorithm == 'cfr+':
                np.maximum(regret, 0., out=regret)
            self.strategy_sum[i] *= gamma
            self.strategy_sum[i] += sigma
            return value
        return sum(self._walk(child, player, opp_reach * sigma[a], discounts)
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
        """Run until ``iterations``, the exploitability target (BB) or the time limit."""
        start = time.perf_counter()
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
                exploit, ev, _ = self.exploitability()
                history.append((self.iterations, spent, exploit))
                if done or target_exploitability is not None and exploit <= target_exploitability:
                    break
        exploit, ev, best = self.exploitability()
        self.stats.update(iterations=self.iterations, exploitability_bb=exploit, value_bb=ev,
                          best_response_values_bb=best, history=history,
                          solve_seconds=time.perf_counter() - start)
        return self

    def hand_values(self):
        """Per-hand on-policy EV (BB) under the average strategy, one array per player.

        Each hand's value is conditional on its own cards: opponent hands that
        share a card are excluded. NaN marks a hand with no compatible opponent.
        """
        out = []
        for p in (0, 1):
            cfv = self._policy_value(self.tree, p, self.weights[p], self.weights[1 - p])
            mass = self._showdown(p, self.weights[1 - p])[3]
            out.append(np.divide(cfv, mass, out=np.full_like(cfv, np.nan), where=mass > 1e-12))
        return out

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
