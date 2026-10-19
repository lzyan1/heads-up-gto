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

    def test_source_examples_and_restricted_jam_mixes(self):
        nodes = self.chart['nodes']
        self.assertEqual(nodes['']['raise:2.5']['T4o'], .2)
        self.assertEqual(nodes['']['raise:2.5']['Q2o'], .6)
        self.assertEqual(nodes['']['raise:2.5']['KJs'], 1)
        self.assertNotIn('raise:100', nodes[''])
        self.assertEqual(nodes['raise:2.5']['call']['J6o'], .4)
        vs3 = nodes['raise:2.5/raise:11.5']
        self.assertEqual(vs3['raise:100']['AKo'], .3)
        self.assertEqual(vs3['raise:100']['99'], .3)
        self.assertEqual(vs3['raise:26']['AKo'], .7)
        self.assertEqual(vs3['raise:26']['99'], .2)
        self.assertEqual(vs3['call']['99'], .5)
        self.assertEqual(vs3['raise:26']['KTo'], .2)
        self.assertEqual(vs3['call']['KTo'], .7)
        self.assertEqual(vs3['raise:26']['AQo'], .5)
        self.assertEqual(vs3['call']['AQo'], .4)
        self.assertNotIn('AKo', vs3['call'])
        self.assertEqual(vs3['raise:26']['TT'], .6)
        self.assertEqual(vs3['call']['TT'], .4)

    def test_short_bars_are_conditional_actions_not_reach_again(self):
        line = reconstruct('raise:2.5/raise:11.5/raise:26/call')
        bb, sb = chart_ranges(self.chart, line, ())
        # Image 4 A8s has a short pure-green bar: BB 3-bets it 30%,
        # then calls 100%. The resulting weight is .3, not .3 squared.
        self.assertAlmostEqual(bb[hand('As8s')], .3)
        self.assertEqual(self.chart['nodes']['raise:2.5/raise:11.5/raise:26']['call']['A8s'], 1)
        self.assertAlmostEqual(bb[hand('KsQs')], .5)
        self.assertAlmostEqual(sb[hand('AsAh')], .9)
        self.assertNotIn(hand('KsKh'), bb)  # KK jams the 4-bet; never calls.

    def test_fifth_screenshot_call_and_fold(self):
        history = ('raise:2.5', 'raise:11.5', 'raise:26', 'raise:100', 'call')
        _, sb = chart_ranges(self.chart, Preflop(history, 200, (0, 0), (100, 100)), ())
        self.assertAlmostEqual(sb[hand('AsAh')], .9)
        self.assertAlmostEqual(sb[hand('TsTh')], .6)
        self.assertNotIn(hand('Ks4s'), sb)
        self.assertNotIn(hand('As5h'), sb)
        with self.assertRaisesRegex(ValueError, 'all-in preflop'):
            prepare({'preflop': '/'.join(history), 'flop': '2c 7d 9h'})

    def test_bb_response_to_sb_four_bet_jam(self):
        node = self.chart['nodes']['raise:2.5/raise:11.5/raise:100']
        for hand_name, call in (('ATs', .2), ('KQs', .8), ('KJs', .1), ('AQo', .9), ('55', .9)):
            self.assertEqual(node['call'][hand_name], call)
            self.assertAlmostEqual(node['fold'][hand_name], 1-call)
        for hand_name in ('AA', 'AKs', 'AKo', 'AQs', 'AJs', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66'):
            self.assertEqual(node['call'][hand_name], 1)
        history = ('raise:2.5', 'raise:11.5', 'raise:100', 'call')
        bb, sb = chart_ranges(self.chart, Preflop(history, 200, (0, 0), (100, 100)), ())
        self.assertAlmostEqual(bb[hand('7s7h')], .6)
        self.assertAlmostEqual(bb[hand('5s5h')], .09)
        self.assertAlmostEqual(sb[hand('AsKh')], .3)
        self.assertNotIn(hand('KsTh'), bb)
        self.assertNotIn('raise:100', self.chart['nodes']['raise:2.5'])
        source = self.chart['provenance']['screenshots']
        self.assertIn('2026-10-19 13:02:54', [x['screenshot'] for x in source])
        self.assertEqual(source[-1]['confirmed_by'], '2026-10-19 13:03:03')

    def test_built_in_is_default_and_supported_pots(self):
        for line, pot, stack in [('raise:2.5/call', 5, 97.5), ('raise:2.5/raise:11.5/call', 23, 88.5), ('raise:2.5/raise:11.5/raise:26/call', 52, 74)]:
            _, _, metadata = prepare({'preflop': line, 'flop': 'As Kd 7c', 'iterations': 10})
            self.assertFalse(metadata['is_demo'])
            self.assertEqual(metadata['range_approximation']['frequency_step'], .1)
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
