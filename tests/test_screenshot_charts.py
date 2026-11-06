import json
from pathlib import Path
import unittest

from heads_up_gto.cards import COMBOS, parse_cards, hand_class
from heads_up_gto.preflop import Preflop, chart_ranges, reconstruct, validate_chart
from heads_up_gto.screenshot_charts import screenshot_chart
from heads_up_gto.server import prepare


def hand(text):
    return tuple(sorted(parse_cards(text, 2)))


class ScreenshotChartsTests(unittest.TestCase):
    def setUp(self):
        self.chart = screenshot_chart()
        self.compiled = validate_chart(self.chart)

    def test_all_six_nodes_and_action_sizes(self):
        self.assertEqual(len(self.compiled), 6)
        histories = list(self.compiled)
        self.assertEqual(histories[-1], 'raise:2.5/raise:11.5/raise:26/raise:100')
        self.assertEqual(set(self.compiled['']), {'fold', 'raise:2.5'})
        self.assertEqual(set(self.compiled[histories[1]]), {'fold', 'call', 'raise:11.5'})
        self.assertEqual(set(self.compiled[histories[2]]), {'fold', 'call', 'raise:26', 'raise:100'})
        self.assertEqual(set(self.compiled[histories[3]]), {'fold', 'call', 'raise:100'})
        self.assertEqual(set(self.compiled[histories[4]]), {'fold', 'call'})
        self.assertEqual(set(self.compiled[histories[5]]), {'fold', 'call'})

    def test_every_visible_mix_totals_one_and_uses_tenths(self):
        for history, actions in self.compiled.items():
            for combo in COMBOS:
                weights = [r.get(combo, 0) for r in actions.values()]
                self.assertAlmostEqual(sum(weights), 0 if hand_class(combo) in self.chart['unobserved_hands'][history] else 1)
                for w in weights:self.assertAlmostEqual(w * 10, round(w * 10))

    def test_all_hands_with_positive_rounded_reach_have_a_strategy(self):
        def walk(prefix, ranges):
            history = '/'.join(prefix)
            actions = self.compiled[history]
            actor = 1 - len(prefix) % 2
            for combo, weight in ranges[actor].items():
                if weight > 0:
                    self.assertAlmostEqual(sum(r.get(combo, 0) for r in actions.values()), 1, msg=f'{history} {hand_class(combo)}')
            for action, chosen in actions.items():
                next_path = [*prefix, action]
                if '/'.join(next_path) in self.compiled:
                    updated = list(ranges)
                    updated[actor] = {h: w * chosen.get(h, 0) for h, w in ranges[actor].items()}
                    walk(next_path, updated)
        walk([], [dict.fromkeys(COMBOS, 1.), dict.fromkeys(COMBOS, 1.)])

    def test_highest_probability_actions_and_removed_fractional_reach(self):
        nodes = self.chart['nodes']
        self.assertEqual(nodes['']['fold']['T4o'], 1)
        self.assertEqual(nodes['']['raise:2.5']['Q2o'], 1)
        vs3 = nodes['raise:2.5/raise:11.5']
        self.assertEqual(vs3['raise:26']['AKo'], 1)
        self.assertEqual(vs3['call']['99'], 1)
        bb, sb = chart_ranges(self.chart, reconstruct('raise:2.5/raise:11.5/raise:26/call'), ())
        self.assertTrue(all(w == 1 for r in (bb, sb) for w in r.values()))
        self.assertNotIn(hand('As8s'), bb)
        self.assertEqual(sb[hand('AsAh')], 1)

    def test_built_in_is_default_and_supported_pots(self):
        for line, pot, stack in [('raise:2.5/call', 5, 97.5), ('raise:2.5/raise:11.5/call', 23, 88.5), ('raise:2.5/raise:11.5/raise:26/call', 52, 74)]:
            _, _, metadata = prepare({'preflop': line, 'flop': 'As Kd 7c', 'iterations': 10})
            self.assertFalse(metadata['is_demo'])
            self.assertEqual(metadata['range_approximation']['frequency_step'], 1.)
            self.assertEqual(metadata['pot'], pot)
            self.assertEqual(metadata['stacks'], (stack, stack))

    def test_unsupported_line_does_not_invent_a_range(self):
        for line in ('call/check', 'raise:2.5/raise:9/call'):
            with self.assertRaisesRegex(ValueError, 'Chart has no frequencies'):
                prepare({'preflop': line, 'flop': 'As Kd 7c'})

    def test_export_matches_hardcoded_source(self):
        path = Path(__file__).parents[1] / 'charts/gto_wizard_screenshots.json'
        self.assertEqual(json.loads(path.read_text()), self.chart)
        self.assertFalse(self.chart['is_demo'])
        self.assertTrue(self.chart['approximate'])
        self.chart['nodes']['']['raise:2.5']['AA'] = 0
        self.assertEqual(screenshot_chart()['nodes']['']['raise:2.5']['AA'], 1)


if __name__ == '__main__':
    unittest.main()
