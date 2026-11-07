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


def _performance_cores():
    """Threads for the compiled engine: performance cores on Apple silicon (efficiency cores slow
    the slowest chunk), otherwise every core."""
    import os
    import subprocess
    import sys
    if sys.platform == 'darwin':
        try:
            return int(subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], capture_output=True,
                                      text=True, timeout=5, check=True).stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return os.cpu_count() or 1


def _numba_available():
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


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
    def __init__(self, root, ranges, settings, algorithm='dcfr', dtype=None, max_bytes=8 * 2 ** 30,
                 engine='auto', threads=None, lazy=False):
        if algorithm not in ('dcfr', 'cfr+'):
            raise ValueError("algorithm must be 'dcfr' or 'cfr+'.")
        if engine not in ('auto', 'numba', 'numpy'):
            raise ValueError("engine must be 'auto', 'numba' or 'numpy'.")
        if engine == 'auto':
            engine = 'numba' if _numba_available() else 'numpy'
        elif engine == 'numba' and not _numba_available():
            raise ImportError("engine='numba' needs numba: pip install numba (or pip install -e '.[fast]').")
        self.engine = engine
        if root.terminal or root.pending or root.street not in (0, 1, 2) or len(root.board) != 3 + root.street:
            raise ValueError('An active decision on the flop, turn or river with a matching board is required.')
        if len(ranges) != 2 or any(not math.isfinite(w) or w < 0 for r in ranges for w in r.values()):
            raise ValueError('Two ranges with nonnegative finite weights are required.')
        start = time.perf_counter()
        self.root, self.settings, self.algorithm = root, settings, algorithm
        self.dtype = np.dtype(dtype if dtype is not None else (np.float32 if engine == 'numba' else np.float64))
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
        live_bytes = sum(c['live'].nbytes for c in self._outcomes.values() if c['live'] is not None)
        prep_bytes = sum(len(b) * n * 4 * (4 if engine == 'numba' else 9) for b in self._prep_boards.values())
        largest = max([d.contexts for d in self.decisions] + [len(b) for b in self._prep_boards.values()] + [1])
        transient = 24 * largest * n * 8 if engine == 'numpy' else 96 * 2 ** 20
        self.memory_bytes = int(self._regret_bytes * item + live_bytes + prep_bytes + transient)
        if self.memory_bytes > max_bytes:
            raise ValueError(
                f'This solve needs about {self.memory_bytes / 2 ** 30:.1f} GiB (limit {max_bytes / 2 ** 30:.1f} GiB). '
                'Narrow the ranges, use dtype=np.float32, shrink the betting tree, or start from a later street.')
        self.iterations = 0
        self.allocated = False
        self._threads = threads
        self._last_check = None  # (iteration, exploitability BB, value BB, best-response values)
        contexts = max([d.contexts for d in self.decisions] + [1])
        self.stats = {'private_hands': n, 'public_decisions': len(self.decisions),
                      'chance_nodes': self._chance_count, 'showdown_leaves': len(self._showdowns),
                      'max_contexts': contexts, 'memory_bytes': self.memory_bytes, 'engine': engine,
                      'build_seconds': time.perf_counter() - start}
        if not lazy:
            self.allocate()

    def allocate(self):
        """Allocate regrets and showdown tables. Called on first use when built with ``lazy=True``,
        so a too-large solve is rejected (and a preview made) without touching memory."""
        if self.allocated:
            return self
        start = time.perf_counter()
        if self.engine == 'numpy':
            self.regret = [np.zeros((len(d.actions), d.contexts, self.size), self.dtype) for d in self.decisions]
            self.strategy_sum = [np.zeros_like(r) for r in self.regret]
        built = self._prepare_showdowns()
        if self.engine == 'numba':
            self._init_numba(built, self._threads)
        self.allocated = True
        self.stats['allocate_seconds'] = time.perf_counter() - start
        return self

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
            live = None
            if self.engine == 'numpy':  # the numba kernels test dealt cards directly
                blocked = self.has[cards[:, 0]].copy()
                for j in range(1, draw):
                    blocked |= self.has[cards[:, j]]
                live = (~blocked).astype(self.dtype)
            found = self._outcomes[key] = dict(
                cards=cards, per_parent=per_parent, live=live,
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

    def _prepare_showdowns_compiled(self):
        """Hands in strength order for every distinct board (numba engine), all boards at once."""
        from . import _postflop_kernels as kernels
        keys = list(self._prep_boards)
        rows = [np.sort(self._prep_boards[k], axis=1) for k in keys]
        unique, inverse = np.unique(np.concatenate(rows), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        strength = kernels.board_strengths(np.ascontiguousarray(unique), self.cards)
        order = np.argsort(strength, axis=1, kind='stable')
        ordered = np.take_along_axis(strength, order, 1)
        group_end = np.stack([np.searchsorted(r, r, 'right') for r in ordered])
        # one row per context: [order | card x | card y | group end], hands in strength order
        packed = np.concatenate((order, self._c1[order], self._c2[order], group_end), axis=1).astype(np.int32)
        built, start = {}, 0
        for key, r in zip(keys, rows):
            built[key] = packed[inverse[start:start + len(r)]]
            start += len(r)
        return built

    def _prepare_showdowns(self):
        if self.engine == 'numba':
            return self._prepare_showdowns_compiled()
        unique, built = {}, {}
        for key, boards in self._prep_boards.items():
            parts = []
            for board in boards:
                b = tuple(sorted(board.tolist()))
                if b not in unique:
                    unique[b] = tuple(a.astype(np.int32) for a in self._board_prep(b))
                parts.append(unique[b])
            built[key] = tuple(np.stack([p[i] for p in parts]) for i in range(len(parts[0])))
        for leaf in self._showdowns:
            leaf.prep = built[leaf.boards.tobytes()]
        return built

    def _init_numba(self, built, threads):
        """Flatten the public tree into the integer tables the compiled kernels walk."""
        import numba
        from . import _postflop_kernels as kernels
        numba.set_num_threads(max(1, min(threads or _performance_cores(), numba.config.NUMBA_NUM_THREADS)))
        n = self.size
        offsets, total = [], 0
        for d in self.decisions:
            offsets.append(total)
            total += d.contexts * len(d.actions) * n
        self._decision_offsets = offsets
        self._regret_flat = np.zeros(total, self.dtype)
        self._strategy_flat = np.zeros(total, self.dtype)

        blocks = []  # one per street betting tree: nodes in preorder, local child indices

        def make_block(top):
            blk = dict(nodes=[], kids=[], chance={})
            blocks.append(blk)
            bid = len(blocks) - 1

            def visit(node):
                local = len(blk['nodes'])
                blk['nodes'].append(node)
                blk['kids'].append(())
                if node.kind == 'decision':
                    blk['kids'][local] = tuple(visit(child) for child in node.children)
                elif node.kind == 'chance':
                    blk['chance'][local] = make_block(node.child)
                return local

            visit(top)
            return bid

        make_block(self.tree)
        block_offset = np.cumsum([0] + [len(b['nodes']) for b in blocks])[:-1].astype(np.int32)
        count = sum(len(b['nodes']) for b in blocks)
        max_actions = max([len(d.actions) for d in self.decisions] + [1])
        ntype = np.zeros(count, np.int8)
        actor = np.zeros(count, np.int8)
        nact = np.zeros(count, np.int32)
        children = np.full((count, max_actions), -1, np.int32)
        reg_off = np.zeros(count, np.int64)
        pay = np.zeros((count, 2, 3))
        ch_blk = np.zeros(count, np.int32)
        ch_m = np.zeros(count, np.int32)
        ch_draw = np.zeros(count, np.int32)
        ch_w = np.zeros(count)
        ch_cards = np.zeros(count, np.int64)
        prep_off = np.zeros(count, np.int64)

        cards_at, cards_parts, cards_total = {}, [], 0
        prep_at, prep_parts, prep_total = {}, [], 0
        kinds = {'decision': 0, 'fold': 1, 'showdown': 2, 'chance': 3}
        for bid, blk in enumerate(blocks):
            for local, node in enumerate(blk['nodes']):
                g = block_offset[bid] + local
                ntype[g] = kinds[node.kind]
                if node.kind == 'decision':
                    actor[g], nact[g] = node.actor, len(node.actions)
                    children[g, :len(node.actions)] = blk['kids'][local]
                    reg_off[g] = offsets[node.index]
                elif node.kind == 'chance':
                    ch_blk[g], ch_m[g], ch_draw[g] = blk['chance'][local], node.per_parent, node.draw
                    ch_w[g] = node.weight
                    if id(node.cards) not in cards_at:
                        cards_at[id(node.cards)] = cards_total
                        cards_parts.append(node.cards.astype(np.int32).ravel())
                        cards_total += node.cards.size
                    ch_cards[g] = cards_at[id(node.cards)]
                else:
                    state = node.state
                    for player in (0, 1):
                        if node.kind == 'fold':
                            pay[g, player, 0] = state.payoff(state.winner, player)
                        else:
                            pay[g, player] = (state.payoff(player, player), state.payoff(1 - player, player),
                                              state.payoff(-1, player))
                    if node.kind == 'showdown':
                        key = node.boards.tobytes()
                        if key not in prep_at:
                            prep_at[key] = prep_total
                            prep_parts.append(built[key].astype(np.int32).ravel())
                            prep_total += built[key].size
                        prep_off[g] = prep_at[key]
        built.clear()

        def depth(bid):
            return 1 + max([depth(child) for child in blocks[bid]['chance'].values()] + [-1])

        if depth(0) > 2:
            raise ValueError('The compiled engine supports at most three nested streets.')
        self._D = 3
        self._maxn = max(len(b['nodes']) for b in blocks)
        self._maxa = max_actions
        c1, c2 = self.cards[:, 0], self.cards[:, 1]
        self._T = (ntype, actor, nact, children, reg_off, pay, ch_blk, ch_m, ch_draw, ch_w, ch_cards,
                   prep_off, block_offset, np.asarray([len(b['nodes']) for b in blocks], np.int32),
                   np.concatenate(cards_parts) if cards_parts else np.zeros(1, np.int32),
                   np.concatenate(prep_parts) if prep_parts else np.zeros(1, np.int32),
                   c1.astype(np.int64), c2.astype(np.int64), self._offset.astype(np.int64))
        units = [(local, j) for local, node in enumerate(blocks[0]['nodes']) if node.kind == 'chance'
                 for j in range(node.per_parent)]
        self._unit_l = np.asarray([u[0] for u in units], np.int64)
        self._unit_j = np.asarray([u[1] for u in units], np.int64)
        self._unit_out = np.zeros((len(units), n), self.dtype)
        self._nchunks = max(1, numba.get_num_threads())  # each chunk allocates its own scratch
        self._kernels = kernels

    def _run(self, player, mode, reach, discounts=(0., 0., 0.)):
        """One compiled root evaluation: 0 train ``player``, 1 best response, 2 average-strategy value."""
        out = np.empty(self.size, self.dtype)
        self._kernels.eval_root(
            self._T, np.ascontiguousarray(reach, dtype=self.dtype), out, player, mode, *map(float, discounts),
            self._regret_flat, self._strategy_flat, self._unit_l, self._unit_j, self._unit_out,
            self._nchunks, self._D, self._maxn, self._maxa)
        return out

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
        self.allocate()
        if self.engine == 'numba':
            best = [float(self.weights[p] @ self._run(p, 1, self.weights[1 - p]).astype(np.float64)) / self.normalizer
                    for p in (0, 1)]
            value = float(self.weights[0] @ self._run(0, 2, self.weights[1]).astype(np.float64)) / self.normalizer
            return max(0., (best[0] + best[1]) / 2), value, best
        reach = [w[None, :] for w in self.weights]
        best = [float(self.weights[p] @ self._best_response(self.tree, p, reach[1 - p])[0]) / self.normalizer
                for p in (0, 1)]
        value = float(self.weights[0] @ self._policy_value(self.tree, 0, reach[1])[0]) / self.normalizer
        return max(0., (best[0] + best[1]) / 2), value, best

    def solve(self, *, iterations=1000, target_exploitability=None, check_every=25, time_limit_seconds=None,
              stop=None, progress=None, check_overhead=None, final_check=True):
        """Run until ``iterations`` more, the exploitability target (BB), the time limit or ``stop()``.

        ``stats['history']`` holds (iteration, solve seconds, exploitability BB); the clock excludes
        exploitability checks. Exploitability is checked every ``check_every`` iterations, or, with
        ``check_overhead`` (a fraction such as 0.15), whenever enough solve time has passed that
        checking costs at most that share of the run: right for iterations that take seconds.
        ``stop`` is polled before every iteration and ``progress(iteration, seconds,
        exploitability or None)`` is called after every one. ``final_check=False`` skips the
        closing exploitability computation (the last periodic check is kept in ``last_check``).
        """
        self.allocate()
        history, spent, steps = [], 0., 0
        last_at, cost = 0., None
        for step in range(iterations):
            if stop is not None and stop():
                break
            began = time.perf_counter()
            discounts = self._discounts(self.iterations)
            self.iterations += 1
            for p in (0, 1):
                if self.engine == 'numba':
                    self._run(p, 0, self.weights[1 - p], discounts)
                else:
                    self._walk(self.tree, p, self._root_reach[1 - p], discounts)
            spent += time.perf_counter() - began
            steps += 1
            done = time_limit_seconds is not None and spent > time_limit_seconds
            if check_overhead is None:
                due = self.iterations % check_every == 0
            else:
                due = steps >= 2 if cost is None else spent - last_at >= cost / check_overhead
            exploit = None
            if due or done or step + 1 == iterations:
                measured = time.perf_counter()
                full = self.exploitability()
                cost, last_at, exploit = time.perf_counter() - measured, spent, full[0]
                history.append((self.iterations, spent, exploit))
                self._last_check = (self.iterations, *full)
                if done or target_exploitability is not None and exploit <= target_exploitability:
                    break
            if progress is not None:
                progress(self.iterations, spent, exploit)
        self.stats.update(iterations=self.iterations, history=history, solve_seconds=spent)
        if final_check and (self._last_check is None or self._last_check[0] != self.iterations):
            self.check()
        if self._last_check is not None:
            _, exploit, value, best = self._last_check
            self.stats.update(exploitability_bb=exploit, value_bb=value, best_response_values_bb=best)
        return self

    def check(self):
        """Compute and record exploitability now: (exploitability, value, best-response values) in BB."""
        full = self.exploitability()
        self._last_check = (self.iterations, *full)
        self.stats.update(exploitability_bb=full[0], value_bb=full[1], best_response_values_bb=full[2])
        return full

    @property
    def last_check(self):
        """(iteration, exploitability BB, value BB, best-response values) of the latest check, or None."""
        return self._last_check

    def hand_values(self):
        """Root per-hand on-policy EV (BB) for each player, as two length-N arrays.

        Conditional on the hand: opponent hands sharing a card are excluded. NaN where
        no compatible opponent exists. Hands outside a player's range get the value of
        playing the average strategy from that hand.
        """
        self.allocate()
        out = []
        for p in (0, 1):
            reach = self.weights[1 - p][None, :]
            if self.engine == 'numba':
                cfv = self._run(p, 2, self.weights[1 - p]).astype(np.float64)
            else:
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
        self.allocate()
        return node.actions, self._average_at(node, k)

    def _average_at(self, node, k):
        if self.engine == 'numpy':
            return self._average(node)[:, k, :]
        a, n = len(node.actions), self.size
        start = self._decision_offsets[node.index] + k * a * n
        s = self._strategy_flat[start:start + a * n].reshape(a, n).astype(np.float64)
        total = s.sum(axis=0)
        return np.where(total > 1e-12, s / np.maximum(total, 1e-20), 1 / a)

    def hand_labels(self):
        return [combo_text(h) for h in self.hands]
