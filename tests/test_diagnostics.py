import unittest
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.exact_river import ExactRiver
from heads_up_gto.game import Settings, State
from heads_up_gto.diagnostics import sample_summary, river_exploitability


class DiagnosticTests(unittest.TestCase):
    def solver(self):
        return ExactRiver(State(2, (3, 3), parse_cards('2c 7d 9h Js Qc'), street=2),
                             (parse_range('AsAh,6s6h'), parse_range('KsKh,5s5h')),
                             Settings(deep_hidden=32, deep_buffer_size=128,
                                      deep_traversals=4, deep_train_steps=2, deep_batch_size=16))

    def test_sample_states_and_threshold_are_separate(self):
        rows = [{'updates': u, 'averages': a, 'mass': .2}
                for u, a in [(0, 0), (0, 2), (3, 0), (1, 900), (200, 200)]]
        summary = sample_summary(rows)
        self.assertTrue(all(v == 1 for v in summary['counts'].values()))
        self.assertAlmostEqual(sum(summary['mass'].values()), 1)
        self.assertEqual(summary['regret_updates']['median'], 1)

    def test_inspection_diagnostics_and_selected_mass(self):
        s = self.solver()
        s.solve()
        from heads_up_gto.river_only import RiverOnlySolver
        adapter = RiverOnlySolver(s.root, s.ranges, s.settings)
        adapter.result = s
        hand = next(iter(s.ranges[0]))
        view = adapter.inspect([], hand)
        self.assertEqual(view['selected'], next(r for r in view['rows'] if r['hand'] == view['selected']['hand']))
        self.assertAlmostEqual(sum(row['mass'] for row in view['rows']), 1)
        self.assertLess(s.stats['exploitability_bb'], 1e-8)

    def test_exact_best_response_scope(self):
        solver = self.solver().solve()
        self.assertEqual(river_exploitability(solver)['hand_pairs'], 4)
        with self.assertRaisesRegex(ValueError, 'Too many hand pairs'):
            river_exploitability(solver, max_pairs=3)
        from dataclasses import replace
        solver.root = replace(solver.root, board=solver.root.board[:4], street=1)
        with self.assertRaisesRegex(ValueError, 'river games'):
            river_exploitability(solver)
