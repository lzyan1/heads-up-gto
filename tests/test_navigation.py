"""Public navigation follows betting rules and preserves chart conditioning."""
import unittest
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State
from heads_up_gto.navigation import preflop_view
from heads_up_gto.preflop import demo_chart, reconstruct
from heads_up_gto.screenshot_charts import screenshot_chart
from heads_up_gto.solver import Solver


class PreflopNavigationTests(unittest.TestCase):
    def setUp(self):
        self.chart = screenshot_chart()

    def test_opening_and_partial_line(self):
        root = preflop_view([], self.chart)
        self.assertEqual(root['actor'], 1)
        self.assertEqual(root['stacks'], [99, 99.5])
        self.assertEqual(len(root['rows']), 1326)
        self.assertEqual(root['coverage'], 1)
        self.assertAlmostEqual(sum(root['aggregate']), 1)
        bb = preflop_view(['raise:2.5'], self.chart)
        self.assertEqual(bb['actor'], 0)
        self.assertEqual(bb['pot'], 3.5)
        self.assertEqual(bb['timeline'][0]['selected'], 'raise:2.5')
        self.assertIsNone(bb['timeline'][1]['selected'])
        self.assertEqual(bb['timeline'][1]['path'], ['raise:2.5'])

    def test_closed_lines_match_solver_reconstruction(self):
        for path in (['raise:2.5', 'call'], ['raise:2.5', 'raise:11.5', 'call'], ['raise:2.5', 'raise:11.5', 'raise:26', 'call']):
            result = preflop_view(path, self.chart)
            expected = reconstruct(path)
            self.assertTrue(result['pending'])
            self.assertFalse(result['terminal'])
            self.assertEqual(result['pot'], expected.pot)
            self.assertEqual(tuple(result['stacks']), expected.stacks)
            self.assertEqual(result['actions'], [])
            self.assertEqual(len(result['timeline']), len(path))

    def test_fold_and_all_in_do_not_request_board(self):
        folded = preflop_view(['raise:2.5', 'fold'], self.chart)
        self.assertEqual(folded['winner'], 1)
        self.assertEqual(folded['terminal'], 'fold')
        self.assertFalse(folded['pending'])
        all_in = preflop_view(['raise:2.5', 'raise:11.5', 'raise:26', 'raise:100', 'call'], self.chart)
        self.assertEqual(all_in['terminal'], 'showdown')
        self.assertFalse(all_in['pending'])
        self.assertEqual(all_in['pot'], 200)
        self.assertEqual(all_in['stacks'], [0, 0])

    def test_missing_branch_is_not_fabricated(self):
        self.chart['nodes'].pop('raise:2.5')
        result = preflop_view(['raise:2.5'], self.chart)
        self.assertTrue(result['missing_chart'])
        self.assertEqual(result['actions'], [])
        with self.assertRaises(ValueError):
            preflop_view(['raise:2.5', 'call'], self.chart)

    def test_conditioning_applied_once(self):
        result = preflop_view(['raise:2.5', 'raise:11.5', 'raise:26'], self.chart)
        a8 = [r for r in result['rows'] if r['class'] == 'A8s']
        self.assertEqual(len(a8), 4)
        call_index = [a['key'] for a in result['actions']].index('call')
        for row in a8:
            self.assertAlmostEqual(row['weight'], .3)
            self.assertEqual(row['probabilities'][call_index], 1)

    def test_incomplete_import_has_no_invented_probability(self):
        chart = screenshot_chart()
        chart['nodes'] = {'': {'raise:2.5': {'AA': .5, 'KK': 1}}}
        result = preflop_view([], chart)
        self.assertTrue(0 < result['coverage'] < .01)
        self.assertIsNone(next(r for r in result['rows'] if r['class'] == 'AA')['probabilities'])
        self.assertEqual(result['aggregate'], [1])

    def test_limp_and_manual_lines(self):
        for chart in (None, demo_chart()):
            limp = preflop_view(['call'], chart)
            self.assertFalse(limp['pending'])
            self.assertEqual(limp['actor'], 0)
            checked = preflop_view(['call', 'check'], chart)
            self.assertTrue(checked['pending'])
            self.assertEqual(checked['pot'], 2)
        self.assertEqual(preflop_view(['raise:3', 'raise:12', 'call'])['pot'], 24)

    def test_invalid_actions_are_rejected(self):
        for path in (['check'], ['raise:1.5'], ['raise:101'], ['raise:2.5', 'check'], ['raise:2.5', 'call', 'check'], ['fold', 'call'], ['call', 'fold']):
            with self.subTest(path=path), self.assertRaises(ValueError):
                preflop_view(path)


class PostflopTimelineTests(unittest.TestCase):
    def setUp(self):
        root = State(5, (97.5, 97.5), parse_cards('2c 7d 9h', 3))
        self.solver = Solver(root, (parse_range('AsAh'), parse_range('KsKh')), Settings(sizes=((.5,),)*3, include_all_in=False))

    def test_timeline_includes_every_decision_and_board(self):
        path = ['check', 'check', 'deal:Ts', 'bet:2.5', 'call', 'deal:3d']
        result = self.solver.inspect(path)
        panels = result['timeline']
        self.assertEqual([p['kind'] for p in panels], ['decision', 'decision', 'board', 'decision', 'decision', 'board', 'decision'])
        self.assertEqual(panels[0]['path'], [])
        self.assertEqual(panels[2]['path'], ['check', 'check'])
        self.assertEqual(panels[3]['path'], path[:3])
        self.assertEqual(panels[-1]['path'], path)
        self.assertEqual(panels[-1]['street'], 'River')
        self.assertEqual(panels[-1]['pot'], 10)
        self.assertEqual(panels[-1]['stack'], 95)
        self.assertIsNone(panels[-1]['selected'])
        self.assertEqual(result['board'], ['2c','7d','9h','Ts','3d'])

    def test_rewind_and_alternate_branch_are_independent(self):
        long = self.solver.inspect(['check', 'check', 'deal:Ts'])
        root = self.solver.inspect([])
        branch = self.solver.inspect(['bet:2.5'])
        self.assertEqual(len(root['timeline']), 1)
        self.assertEqual(root['timeline'][0]['selected'], None)
        self.assertEqual(branch['timeline'][0]['selected'], 'bet:2.5')
        self.assertEqual(branch['pot'], 7.5)
        self.assertEqual(long['timeline'][0]['selected'], 'check')

    def test_pending_terminal_and_duplicate_card(self):
        pending = self.solver.inspect(['check','check'])
        self.assertTrue(pending['pending'])
        self.assertEqual(len(pending['timeline']), 2)
        folded = self.solver.inspect(['bet:2.5','fold'])
        self.assertEqual(folded['terminal'], 'fold')
        self.assertEqual(len(folded['timeline']), 2)
        with self.assertRaises(ValueError):
            self.solver.inspect(['check','check','deal:2c'])
        river = self.solver.inspect(['check','check','deal:Ts','check','check','deal:3d','check','check'])
        self.assertEqual(river['terminal'], 'showdown')
        self.assertFalse(river['pending'])
