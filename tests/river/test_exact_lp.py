"""Numerical certificates plus an independently known Kuhn-in-Hold'em answer."""
import importlib.util
import itertools
import unittest
from dataclasses import replace

from heads_up_gto.cards import parse_cards, parse_range, score
from heads_up_gto.game import Settings, State

HAS_SCIPY = importlib.util.find_spec('scipy') is not None


def kuhn_game():
    # Same exact combo in both ranges: same-rank deals collide, different ranks
    # are disjoint. This gives precisely six equiprobable ordered private deals.
    root = State(2, (1, 1), parse_cards('2c 5d 7h 9s Jc'), street=2)
    hands = parse_range('As3s,Kh3h,Qd3d')
    settings = Settings(sizes=((), (), (.5,)), max_raises=0)
    return root, (hands, hands), settings


@unittest.skipUnless(HAS_SCIPY, 'Install project dependencies to validate the LP solver')
class ExactRiverTests(unittest.TestCase):
    def make(self):
        from heads_up_gto.river.exact_lp import ExactRiver
        return ExactRiver(*kuhn_game()).solve(tolerance=1e-10)

    def test_kuhn_deal_distribution_value_and_unique_player_two_mixes(self):
        solver = self.make()
        self.assertEqual(solver.stats['legal_pairs'], 6)
        self.assertAlmostEqual(solver.stats['value_bb'], -1 / 18, places=11)
        self.assertLess(solver.stats['exploitability_bb'], 1e-10)
        self.assertLess(abs(solver.stats['lp_gap_bb']), 1e-10)
        root = solver.root
        check = root.apply(root.actions(solver.settings)[0])
        bet = root.apply(root.actions(solver.settings)[1])
        for name, after_check, facing_bet in [('As3s', [0, 1], [0, 1]),
                                             ('Kh3h', [1, 0], [2/3, 1/3]),
                                             ('Qd3d', [2/3, 1/3], [1, 0])]:
            hand = next(iter(parse_range(name)))
            for state, expected in ((check, after_check), (bet, facing_bet)):
                for actual, value in zip(solver.policy(state, hand)[0], expected):
                    self.assertAlmostEqual(actual, value, places=10)
        ranks = [score(h, root.board) for h in solver.hands[0]]
        self.assertTrue(all(a > b for a, b in zip(ranks, ranks[1:])))
        self.assertTrue(all(a[0] == 0 for a in ranks))

    def test_per_hand_continuation_values_integrate_opponent_cards(self):
        solver = self.make()
        for text, value in [('As3s', 7/6), ('Kh3h', -1/3), ('Qd3d', -1)]:
            hand = next(iter(parse_range(text)))
            self.assertAlmostEqual(solver.value_for_hand(0, hand), value, places=10)
        solver.compact()
        self.assertAlmostEqual(solver.value_for_hand(0, next(iter(parse_range('As3s')))), 7/6, places=10)

    def test_independent_normal_form_kuhn_payoff_matrix(self):
        # Enumerate 64 complete deterministic policies per player, using only
        # private rank and observable history. No sequence-form code is reused.
        import numpy as np
        from scipy.optimize import linprog
        solver = self.make()
        matrix = np.zeros((64, 64))
        plans = list(itertools.product((0, 1), repeat=6))
        for i, p0 in enumerate(plans):
            for j, p1 in enumerate(plans):
                def walk(state, h0, h1):
                    if state.terminal:
                        winner = state.winner if state.terminal == 'fold' else 0 if h0 < h1 else 1
                        return state.payoff(winner, 0)
                    h = h0 if state.actor == 0 else h1
                    stage = int(bool(state.path)) if state.actor == 0 else int(state.path == ('bet:1',))
                    action = (p0 if state.actor == 0 else p1)[2 * h + stage]
                    return walk(state.apply(state.actions(solver.settings)[action]), h0, h1)
                matrix[i, j] = sum(walk(solver.root, a, b) for a, b in itertools.permutations(range(3), 2)) / 6
        # All-pure saddle points need not exist; the minimax LP remains solvable.
        self.assertGreater(np.min(np.max(matrix, axis=0)) - np.max(np.min(matrix, axis=1)), .1)
        lp = linprog(np.r_[np.zeros(64), -1],
                     A_ub=np.column_stack((-matrix.T, np.ones(64))), b_ub=np.zeros(64),
                     A_eq=np.array([[*np.ones(64), 0]]), b_eq=[1],
                     bounds=[(0, None)] * 64 + [(None, None)], method='highs')
        self.assertTrue(lp.success)
        self.assertAlmostEqual(-lp.fun, solver.stats['value_bb'], places=10)
        histories = [((), ('check', 'bet:1')), (('check',), ('bet:1',))]
        mixes = []
        for player in (0, 1):
            probabilities = []
            for plan in plans:
                probability = 1.
                for hand_index in range(3):
                    for stage, path in enumerate(histories[player]):
                        probability *= solver.policies[path][hand_index, plan[2 * hand_index + stage]]
                probabilities.append(probability)
            mixes.append(np.asarray(probabilities))
        self.assertAlmostEqual(float(mixes[0].sum()), 1)
        self.assertAlmostEqual(float(mixes[1].sum()), 1)
        self.assertLess(float(np.max(matrix @ mixes[1]) - np.min(mixes[0] @ matrix)), 1e-10)

    def test_weighted_ranges_and_recursive_certificate_match_existing_diagnostic(self):
        from heads_up_gto.river.exact_lp import ExactRiver
        from heads_up_gto.diagnostics import river_exploitability
        root = State(21, (89.5, 89.5), parse_cards('As Kd 7c 2h 3d'), street=2,
                     sizing_profile='single_raised', aggressor_history=(1, 1))
        ranges = parse_range('AhAd:.3,QhQd,JhJd:.8,6s6h'), parse_range('AcAd:.7,QsQh,TsTh:.4,5s5h')
        solver = ExactRiver(root, ranges, Settings()).solve(tolerance=1e-9)
        self.assertEqual([a.key for a in solver.tree.actions], ['check', 'bet:10.5'])
        after_check = solver.nodes[('check',)]
        self.assertEqual([a.key for a in after_check.actions], ['check', 'bet:6.9', 'bet:21'])
        for bet in ('bet:6.9', 'bet:21'):
            self.assertEqual([a.key for a in solver.nodes[('check', bet)].actions],
                             ['fold', 'call', 'raise:89.5'])
        diagnostic = river_exploitability(solver)
        self.assertLess(diagnostic['exploitability_bb'], 1e-9)
        self.assertAlmostEqual(diagnostic['exploitability_bb'], solver.stats['exploitability_bb'], places=10)
        self.assertGreater(solver.stats['terminal_histories'], 5)
        for values in solver.policies.values():
            for row in values:
                self.assertAlmostEqual(sum(row), 1)
                self.assertTrue(all(0 <= v <= 1 for v in row))

    def test_invalid_roots_ranges_and_explicit_memory_guard(self):
        from heads_up_gto.river.exact_lp import ExactRiver
        root, ranges, settings = kuhn_game()
        for state in (replace(root, street=1), replace(root, pending=True), replace(root, terminal='showdown')):
            with self.assertRaises(ValueError):
                ExactRiver(state, ranges, settings)
        for rr in (({}, ranges[0]), (parse_range('AsAh'), parse_range('AsAd'))):
            with self.assertRaises(ValueError):
                ExactRiver(root, rr, settings)
        with self.assertRaisesRegex(ValueError, 'guard'):
            ExactRiver(root, ranges, settings, max_payoff_entries=1)
