"""Full-width vectorized DCFR over the whole postflop game (flop, turn and river).

No sampling and no neural networks. The public tree comes from ``State``: betting
decisions, folds, showdowns, and chance nodes where a street ends. Every private
hand is one element of a numpy array, and every possible dealt card is one
element of a leading *context* axis, so one tree node holds arrays shaped
``[contexts, hands]`` and one traversal updates every hand in every runout at
once.

Contexts. The betting tree under each dealt card is identical (sizing is board
independent), so it is stored once. At a chance node with ``K`` parent contexts
and ``m`` possible next cards the child has ``K * m`` contexts, grouped by
parent. A river decision reached from a flop root therefore holds
``49 * 48`` contexts of regrets. Hands that hold a dealt card get zero reach in
that branch (``live`` masks), and their own values are zeroed on the way back up.

Chance. After a street ends the next card is uniform over the ``52 - b - 4``
cards not on the board (``b`` cards) and not in either hole hand. Using a
constant ``1 / (52 - b - 4)`` together with the live masks gives exactly that
probability for every compatible pair of hands. An all-in before the river is a
chance node that deals every remaining card set at once (unordered, weight
``1 / C(52 - b - 4, k)``) in front of a showdown leaf.

Leaves. Folds use inclusion-exclusion over the two cards each hand holds.
Showdowns sort hands by strength per board and use prefix sums, so a leaf costs
O(N) per context instead of O(N^2): reach mass that wins, loses, ties or is
compatible for every own hand, with card removal by inclusion-exclusion.

Algorithm. ``'dcfr'`` follows b-inary/postflop-solver (regret discounts
t^1.5/(t^1.5+1) when positive and 0.5 when negative, cumulative strategy
discounted by (t'/(t'+1))^3 and restarted at powers of four, not weighted by own
reach). ``'cfr+'`` floors regrets at zero. Players alternate traversals.

Memory is the practical limit: regrets and strategy sums for every river
decision are stored for every runout. ``memory_bytes`` is computed before any
large allocation and compared with ``max_bytes``.
"""
import itertools
import math
import time

import numpy as np

from .cards import combo_text, parse_cards, remove_blocked, score


class _Decision:
    kind = 'decision'
    __slots__ = ('state', 'actor', 'actions', 'children', 'index', 'contexts')


class _Chance:
    kind = 'chance'
    __slots__ = ('parent_contexts', 'per_parent', 'weight', 'live', 'cards', 'child', 'draw')


class _Fold:
    kind = 'fold'
    __slots__ = ('state',)


class _Showdown:
    kind = 'showdown'
    __slots__ = ('state', 'boards', 'prep')


class PostflopCFR:
    def __init__(self, root, ranges, settings, algorithm='dcfr', dtype=np.float64, max_bytes=8 * 2 ** 30):
        if algorithm not in ('dcfr', 'cfr+'):
            raise ValueError("algorithm must be 'dcfr' or 'cfr+'.")
        if root.terminal or root.pending or root.street not in (0, 1, 2) or len(root.board) != 3 + root.street:
            raise ValueError('An active decision on the flop, turn or river with a matching board is required.')
        if len(ranges) != 2 or any(not math.isfinite(w) or w < 0 for r in ranges for w in r.values()):
            raise ValueError('Two ranges with nonnegative finite weights are required.')
        start = time.perf_counter()
        self.root, self.settings, self.algorithm = root, settings, algorithm
        self.dtype = np.dtype(dtype)
        ranges = [remove_blocked(r, root.board) for r in ranges]
        if any(not r for r in ranges):
            raise ValueError('Both players need an unblocked range.')
        self.hands = sorted({tuple(sorted(h)) for r in ranges for h in r})
        index = {h: i for i, h in enumerate(self.hands)}
        n = self.size = len(self.hands)
        weights = np.zeros((2, n))
        for p, r in enumerate(ranges):
            for h, w in r.items():
                weights[p, index[tuple(sorted(h))]] = w
        self.weights = weights / weights.sum(axis=1, keepdims=True)
        self._root_reach = [w[None, :].astype(self.dtype) for w in self.weights]

        # Card geometry shared by every context.
        self.cards = np.asarray(self.hands, dtype=np.int64)
        self.has = np.zeros((52, n), dtype=bool)
        self.has[self.cards[:, 0], np.arange(n)] = True
        self.has[self.cards[:, 1], np.arange(n)] = True
        self.card_matrix = self.has.T.astype(np.float64)
        self._offset = np.concatenate(([0], np.cumsum(self.has.sum(axis=1))))
        c1, c2 = self.cards[:, 0], self.cards[:, 1]
        self._c1, self._c2 = c1, c2
        self._s1, self._s2 = self._offset[c1], self._offset[c2]
        self._e1, self._e2 = self._offset[c1 + 1], self._offset[c2 + 1]

        self.normalizer = float(self.weights[0] @ self._legal(self.weights[1][None, :])[0])
        if self.normalizer <= 0:
            raise ValueError('The ranges have no compatible private deal.')

        # Public tree. Structure first (cheap); big allocations after the memory check.
        self.decisions, self._showdowns = [], []
        self._outcomes, self._chance_count = {}, 0
        self._regret_bytes = 0
        self.tree = self._build(root, np.asarray([root.board], dtype=np.int64))
        self._prep_boards = {}
        for leaf in self._showdowns:
            self._prep_boards.setdefault(leaf.boards.tobytes(), leaf.boards)
        item = self.dtype.itemsize
        live_bytes = sum(c['live'].nbytes for c in self._outcomes.values())
        prep_bytes = sum(len(b) * n * 4 * 9 for b in self._prep_boards.values())
        largest = max([d.contexts for d in self.decisions] + [len(b) for b in self._prep_boards.values()] + [1])
        self.memory_bytes = int(self._regret_bytes * item + live_bytes + prep_bytes + 24 * largest * n * 8)
        if self.memory_bytes > max_bytes:
            raise ValueError(
                f'This solve needs about {self.memory_bytes / 2 ** 30:.1f} GiB (limit {max_bytes / 2 ** 30:.1f} GiB). '
                'Narrow the ranges, use dtype=np.float32, shrink the betting tree, or start from a later street.')
        self.regret = [np.zeros((len(d.actions), d.contexts, n), self.dtype) for d in self.decisions]
        self.strategy_sum = [np.zeros_like(r) for r in self.regret]
        self._prepare_showdowns()
        self.iterations = 0
        contexts = max([d.contexts for d in self.decisions] + [1])
        self.stats = {'private_hands': n, 'public_decisions': len(self.decisions),
                      'chance_nodes': self._chance_count, 'showdown_leaves': len(self._showdowns),
                      'max_contexts': contexts, 'memory_bytes': self.memory_bytes,
                      'build_seconds': time.perf_counter() - start}

    # ------------------------------------------------------------------ tree
    def _deal(self, boards, draw):
        """Every way to deal ``draw`` more cards from each parent context, grouped by parent."""
        key = (boards.tobytes(), draw)
        found = self._outcomes.get(key)
        if found is None:
            rows = []
            for board in boards:
                used = set(board.tolist())
                rows.append(np.asarray(list(itertools.combinations([c for c in range(52) if c not in used], draw)),
                                       dtype=np.int64))
            cards = np.concatenate(rows)
            per_parent = len(rows[0])
            blocked = self.has[cards[:, 0]].copy()
            for j in range(1, draw):
                blocked |= self.has[cards[:, j]]
            found = self._outcomes[key] = dict(
                cards=cards, per_parent=per_parent, live=(~blocked).astype(self.dtype),
                boards=np.concatenate((np.repeat(boards, per_parent, axis=0), cards), axis=1))
        return found

    def _chance(self, boards, draw, make_child):
        out = self._deal(boards, draw)
        node = _Chance()
        node.parent_contexts, node.per_parent, node.draw = len(boards), out['per_parent'], draw
        node.live, node.cards = out['live'], out['cards']
        node.weight = 1. / math.comb(52 - boards.shape[1] - 4, draw)
        node.child = make_child(out['boards'])
        self._chance_count += 1
        return node

    def _build(self, state, boards):
        if state.terminal == 'fold':
            node = _Fold()
            node.state = state
            return node
        if state.terminal == 'showdown':
            if boards.shape[1] == 5:
                node = _Showdown()
                node.state, node.boards, node.prep = state, boards, None
                self._showdowns.append(node)
                return node
            # All-in before the river: deal every remaining card set, then show down.
            return self._chance(boards, 5 - boards.shape[1], lambda b: self._build(state, b))
        if state.pending:
            spare = next(c for c in range(52) if c not in state.board)
            child = state.deal(spare)  # betting structure does not depend on which card
            return self._chance(boards, 1, lambda b: self._build(child, b))
        node = _Decision()
        node.state, node.actor, node.contexts = state, state.actor, len(boards)
        node.actions = state.actions(self.settings)
        node.index = len(self.decisions)
        self.decisions.append(node)
        self._regret_bytes += 2 * len(node.actions) * len(boards) * self.size
        node.children = tuple(self._build(state.apply(a), boards) for a in node.actions)
        return node

    # ------------------------------------------------------------- showdowns
    def _board_prep(self, board):
        """Per-board sort order and prefix-sum indices for one 5-card board."""
        n = self.size
        blocked = self.has[list(board)].any(axis=0)
        alive = np.flatnonzero(~blocked)
        scores = [score(self.hands[i], board) for i in alive]
        rank = {s: i + 1 for i, s in enumerate(sorted(set(scores)))}
        strength = np.zeros(n, dtype=np.int64)  # board-blocked hands sort first with zero reach
        strength[alive] = [rank[s] for s in scores]
        order = np.argsort(strength, kind='stable')
        ordered = strength[order]
        lo = np.searchsorted(ordered, strength, 'left')
        hi = np.searchsorted(ordered, strength, 'right')
        member = np.zeros((52, n), dtype=bool)
        position = np.arange(n)
        member[self.cards[order, 0], position] = True
        member[self.cards[order, 1], position] = True
        count = np.zeros((52, n + 1), dtype=np.int64)
        np.cumsum(member, axis=1, out=count[:, 1:])
        _, columns = np.nonzero(member)  # holders of card 0, then card 1, ... in sorted order
        c1, c2 = self._c1, self._c2
        return (order, order[columns], lo, hi, self._s1 + count[c1, lo], self._s2 + count[c2, lo],
                self._s1 + count[c1, hi], self._s2 + count[c2, hi])

    def _prepare_showdowns(self):
        unique, built = {}, {}
        for key, boards in self._prep_boards.items():
            parts = []
            for board in boards:
                b = tuple(sorted(board.tolist()))
                if b not in unique:
                    unique[b] = tuple(a.astype(np.int32) for a in self._board_prep(b))
                parts.append(unique[b])
            built[key] = tuple(np.stack([p[i] for p in parts]) for i in range(8))
        for leaf in self._showdowns:
            leaf.prep = built[leaf.boards.tobytes()]

    def _legal(self, reach):
        """Opponent reach compatible with each own hand (no shared card), ``[K, N]``."""
        r = reach.astype(np.float64, copy=False)
        by_card = r @ self.card_matrix
        return r.sum(axis=1, keepdims=True) - by_card[:, self._c1] - by_card[:, self._c2] + r

    def _rows(self, prep, reach):
        """Win, loss, tie and compatible opponent reach per own hand and context."""
        order, flat, lo, hi, a1, a2, b1, b2 = prep
        r = reach.astype(np.float64, copy=False)
        k, n = r.shape
        take = np.take_along_axis
        prefix = np.zeros((k, n + 1))
        np.cumsum(take(r, order, 1), axis=1, out=prefix[:, 1:])
        held = np.zeros((k, flat.shape[1] + 1))
        np.cumsum(take(r, flat, 1), axis=1, out=held[:, 1:])
        base = held[:, self._s1] + held[:, self._s2]
        win = take(prefix, lo, 1) - (take(held, a1, 1) + take(held, a2, 1) - base)
        upto = take(prefix, hi, 1) - (take(held, b1, 1) + take(held, b2, 1) - base) + r
        legal = prefix[:, -1:] - (held[:, self._e1] + held[:, self._e2] - base) + r
        return win, legal - upto, upto - win, legal

    def _leaf(self, node, player, reach):
        state = node.state
        if node.kind == 'fold':
            out = state.payoff(state.winner, player) * self._legal(reach)
        else:
            win, loss, tie, _ = self._rows(node.prep, reach)
            out = (state.payoff(player, player) * win + state.payoff(1 - player, player) * loss
                   + state.payoff(-1, player) * tie)
        return out.astype(self.dtype, copy=False)

    # -------------------------------------------------------------- strategy
    def _current(self, node):
        positive = np.maximum(self.regret[node.index], 0.)
        total = positive.sum(axis=0)
        return np.where(total > 1e-12, positive / np.maximum(total, 1e-20), 1 / len(node.actions))

    def _average(self, node):
        s = self.strategy_sum[node.index].astype(np.float64)
        total = s.sum(axis=0)
        return np.where(total > 1e-12, s / np.maximum(total, 1e-20), 1 / len(node.actions))

    def _discounts(self, t):
        """(alpha, beta, gamma) for 0-based iteration ``t``."""
        if self.algorithm == 'cfr+':
            return 1., 0., (t + 1.) / (t + 2.)
        lower = 0 if t == 0 else 1 << ((t.bit_length() - 1) & ~1)  # 0, 1, 4, 16, ...
        a = max(t - 1, 0) ** 1.5
        g = t - lower
        return a / (a + 1.), .5, (g / (g + 1.)) ** 3

    @staticmethod
    def _down(node, reach):
        return np.repeat(reach, node.per_parent, axis=0) * node.live

    @staticmethod
    def _up(node, values):
        summed = (values * node.live).reshape(node.parent_contexts, node.per_parent, -1).sum(axis=1)
        return summed * node.weight

    def _walk(self, node, player, reach, discounts):
        """One DCFR traversal for ``player``; returns counterfactual values ``[K, N]``."""
        if node.kind == 'chance':
            return self._up(node, self._walk(node.child, player, self._down(node, reach), discounts))
        if node.kind != 'decision':
            return self._leaf(node, player, reach)
        sigma = self._current(node)
        if node.actor != player:
            total = None
            for a, child in enumerate(node.children):
                v = self._walk(child, player, reach * sigma[a], discounts)
                total = v if total is None else total + v
            return total
        values = np.empty_like(sigma)
        for a, child in enumerate(node.children):
            values[a] = self._walk(child, player, reach, discounts)
        value = (sigma * values).sum(axis=0)
        alpha, beta, gamma = discounts
        regret = self.regret[node.index]
        regret *= np.where(regret >= 0, alpha, beta)
        regret += values - value
        if self.algorithm == 'cfr+':
            np.maximum(regret, 0., out=regret)
        self.strategy_sum[node.index] *= gamma
        self.strategy_sum[node.index] += sigma
        return value

    def _best_response(self, node, player, reach):
        if node.kind == 'chance':
            return self._up(node, self._best_response(node.child, player, self._down(node, reach)))
        if node.kind != 'decision':
            return self._leaf(node, player, reach)
        if node.actor == player:
            return np.max([self._best_response(c, player, reach) for c in node.children], axis=0)
        sigma = self._average(node)
        return sum(self._best_response(c, player, reach * sigma[a]) for a, c in enumerate(node.children))

    def _policy_value(self, node, player, reach):
        if node.kind == 'chance':
            return self._up(node, self._policy_value(node.child, player, self._down(node, reach)))
        if node.kind != 'decision':
            return self._leaf(node, player, reach)
        sigma = self._average(node)
        if node.actor == player:
            return sum(sigma[a] * self._policy_value(c, player, reach) for a, c in enumerate(node.children))
        return sum(self._policy_value(c, player, reach * sigma[a]) for a, c in enumerate(node.children))

    # ---------------------------------------------------------------- public
    def exploitability(self):
        """(mean best-response gain, player-0 value, best-response values), all in BB.

        Evaluated on the average strategy.
        """
        reach = [w[None, :] for w in self.weights]
        best = [float(self.weights[p] @ self._best_response(self.tree, p, reach[1 - p])[0]) / self.normalizer
                for p in (0, 1)]
        value = float(self.weights[0] @ self._policy_value(self.tree, 0, reach[1])[0]) / self.normalizer
        return max(0., (best[0] + best[1]) / 2), value, best

    def solve(self, *, iterations=1000, target_exploitability=None, check_every=25, time_limit_seconds=None):
        """Run until ``iterations`` more, the exploitability target (BB) or the time limit.

        ``stats['history']`` holds (iteration, solve seconds, exploitability BB); the clock
        excludes exploitability checks.
        """
        history, spent = [], 0.
        for step in range(iterations):
            began = time.perf_counter()
            discounts = self._discounts(self.iterations)
            self.iterations += 1
            for p in (0, 1):
                self._walk(self.tree, p, self._root_reach[1 - p], discounts)
            spent += time.perf_counter() - began
            done = time_limit_seconds is not None and spent > time_limit_seconds
            if self.iterations % check_every == 0 or done or step + 1 == iterations:
                exploit = self.exploitability()[0]
                history.append((self.iterations, spent, exploit))
                if done or target_exploitability is not None and exploit <= target_exploitability:
                    break
        exploit, value, best = self.exploitability()
        self.stats.update(iterations=self.iterations, exploitability_bb=exploit, value_bb=value,
                          best_response_values_bb=best, history=history, solve_seconds=spent)
        return self

    def hand_values(self):
        """Root per-hand on-policy EV (BB) for each player, as two length-N arrays.

        Conditional on the hand: opponent hands sharing a card are excluded. NaN where
        no compatible opponent exists. Hands outside a player's range get the value of
        playing the average strategy from that hand.
        """
        out = []
        for p in (0, 1):
            reach = self.weights[1 - p][None, :]
            cfv = self._policy_value(self.tree, p, reach)[0]
            mass = self._legal(reach)[0]
            out.append(np.divide(cfv, mass, out=np.full(cfv.shape, np.nan), where=mass > 1e-12))
        return out

    def locate(self, path):
        """Follow action keys and dealt cards, e.g. ``['check', 'bet:5', 'call', 'Qc']``.

        Returns the node and the context index reached.
        """
        node, k = self.tree, 0
        for token in path:
            if node.kind == 'chance':
                if node.draw > 1:
                    raise ValueError('Runouts after an all-in deal several cards and have no decisions.')
                card = parse_cards(token, 1)[0]
                rows = node.cards[k * node.per_parent:(k + 1) * node.per_parent, 0]
                k = k * node.per_parent + int(np.flatnonzero(rows == card)[0])
                node = node.child
            elif node.kind == 'decision':
                keys = [a.key for a in node.actions]
                if token not in keys:
                    raise ValueError(f'{token!r} is not available here; choose from {keys}.')
                node = node.children[keys.index(token)]
            else:
                raise ValueError('The path continues past a terminal node.')
        return node, k

    def strategy(self, path=()):
        """Average strategy at a decision: (actions, ``[actions, hands]`` probabilities).

        A pending street end needs the next card in ``path``; hands holding a dealt card are
        meaningless there.
        """
        node, k = self.locate(path)
        if node.kind != 'decision':
            raise ValueError('Not a decision node; add the dealt card or action to the path.')
        return node.actions, self._average(node)[:, k, :]

    def hand_labels(self):
        return [combo_text(h) for h in self.hands]
