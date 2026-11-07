"""Full-tree DCFR: chance nodes and runouts against brute force, river parity, convergence."""
from itertools import combinations
import random
import unittest

import numpy as np

from heads_up_gto.cards import COMBOS, parse_cards, remove_blocked, score
from heads_up_gto.cfr_plus_river import CFRPlusRiver
from heads_up_gto.game import Settings, State
from heads_up_gto.postflop_cfr import PostflopCFR

TREE = dict(raise_multiplier=1e6, max_raises=2, include_all_in=True)  # raises are all-in only


def random_ranges(board, sizes, seed):
    pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), board))
    rng = random.Random(seed)
    return [{h: rng.uniform(.1, 1.) for h in rng.sample(pool, n)} for n in sizes]


def reference_uniform_value(root, ranges, settings):
    """Player-0 value when both players mix uniformly, by explicit enumeration of every
    pair of hands and every dealt card. Independent of all of the solver's array code."""
    def ev(state, hands):
        if state.terminal == 'fold':
            return state.payoff(state.winner, 0)
        if state.terminal == 'showdown' or state.pending:
            used = {*state.board, *hands[0], *hands[1]}
            deck = [c for c in range(52) if c not in used]
            if state.pending:
                return sum(ev(state.deal(c), hands) for c in deck) / len(deck)
            missing = 5 - len(state.board)
            total = count = 0
            for extra in combinations(deck, missing):
                board = (*state.board, *extra)
                a, b = score(hands[0], board), score(hands[1], board)
                total += state.payoff(0 if a > b else 1 if a < b else -1, 0)
                count += 1
            return total / count
        actions = state.actions(settings)
        return sum(ev(state.apply(a), hands) for a in actions) / len(actions)

    num = den = 0.
    for h0, w0 in ranges[0].items():
        for h1, w1 in ranges[1].items():
            if set(h0) & set(h1):
                continue
            num += w0 * w1 * ev(root, (h0, h1))
            den += w0 * w1
    return num / den


class PostflopCFRTests(unittest.TestCase):
    def check_uniform(self, root, settings, sizes, seed):
        ranges = random_ranges(root.board, sizes, seed)
        solver = PostflopCFR(root, ranges, settings)
        value = solver.exploitability()[1]  # empty strategy sums = uniform play
        self.assertAlmostEqual(value, reference_uniform_value(root, ranges, settings), places=9)

    def test_turn_chance_matches_brute_force(self):
        root = State(12, (30, 30), parse_cards('As Kd 7c 2h'), street=1)
        self.check_uniform(root, Settings(sizes=((), (.5,), (1.,)), **TREE), (4, 5), 1)

    def test_flop_chance_and_all_in_runouts_match_brute_force(self):
        # Short stacks: bets and shoves put players all-in on the flop and turn, so this
        # exercises two-card and one-card runouts as well as ordinary street changes.
        root = State(10, (3, 3), parse_cards('As Kd 7c'), street=0)
        self.check_uniform(root, Settings(sizes=((.5,), (.5,), (.5,)), **TREE), (3, 4), 2)

    def test_blockers_between_hands_and_dealt_cards(self):
        # Identical combos in both ranges plus cards that are dealt later.
        root = State(8, (4, 4), parse_cards('Qh 9d 4s 2c'), street=1)
        ranges = random_ranges(root.board, (6, 6), 5)
        ranges[1].update(dict(list(ranges[0].items())[:3]))
        solver = PostflopCFR(root, ranges, Settings(sizes=((), (.5,), (1.,)), **TREE))
        ref = reference_uniform_value(root, ranges, Settings(sizes=((), (.5,), (1.,)), **TREE))
        self.assertAlmostEqual(solver.exploitability()[1], ref, places=9)

    def test_river_root_matches_single_river_solver(self):
        settings = Settings(sizes=((), (), (.33, 1.)), **TREE)
        root = State(21, (89.5, 89.5), parse_cards('As Kd 7c 2h 3d'), street=2)
        ranges = random_ranges(root.board, (60, 60), 3)
        full = PostflopCFR(root, ranges, settings).solve(iterations=40, check_every=1000)
        single = CFRPlusRiver(root, ranges, settings).solve(iterations=40, check_every=1000)
        self.assertAlmostEqual(full.stats['value_bb'], single.stats['value_bb'], places=8)
        self.assertAlmostEqual(full.stats['exploitability_bb'], single.stats['exploitability_bb'], places=8)

    def test_turn_solve_converges_and_strategy_is_a_distribution(self):
        settings = Settings(sizes=((), (.5,), (1.,)), **TREE)
        root = State(21, (60, 60), parse_cards('As Kd 7c 2h'), street=1)
        solver = PostflopCFR(root, random_ranges(root.board, (25, 25), 4), settings)
        solver.solve(iterations=600, target_exploitability=.01 * 21)
        self.assertLess(solver.stats['exploitability_bb'], .01 * 21)
        actions, probs = solver.strategy([])
        np.testing.assert_allclose(probs.sum(axis=0), 1., atol=1e-9)
        # Walk down an action, then past the street change by naming the river card.
        later = solver.strategy(['check', 'check', '9s'])[1]
        np.testing.assert_allclose(later.sum(axis=0), 1., atol=1e-9)

    def test_flop_solve_decreases_exploitability(self):
        settings = Settings(sizes=((.5,), (.5,), (.5,)), **TREE)
        root = State(10, (3, 3), parse_cards('As Kd 7c'), street=0)
        solver = PostflopCFR(root, random_ranges(root.board, (6, 6), 6), settings)
        solver.solve(iterations=200, check_every=20)
        history = [e for _, _, e in solver.stats['history']]
        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], .02 * root.pot)

    def test_memory_guard_and_input_validation(self):
        settings = Settings(sizes=((), (.5,), (1.,)), **TREE)
        root = State(21, (60, 60), parse_cards('As Kd 7c 2h'), street=1)
        ranges = random_ranges(root.board, (20, 20), 7)
        with self.assertRaisesRegex(ValueError, 'needs about'):
            PostflopCFR(root, ranges, settings, max_bytes=1000)
        with self.assertRaises(ValueError):
            PostflopCFR(State(21, (60, 60), parse_cards('As Kd 7c'), street=1), ranges, settings)


if __name__ == '__main__':
    unittest.main()
