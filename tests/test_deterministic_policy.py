import unittest
import numpy as np
from heads_up_gto.deep_cfr import greedy_policy
from heads_up_gto.screenshot_charts import screenshot_chart
from heads_up_gto.preflop import validate_chart
from heads_up_gto.cards import COMBOS, hand_class


class DeterministicPolicyTests(unittest.TestCase):
    def test_each_observed_preflop_hand_has_exactly_one_action(self):
        chart = screenshot_chart()
        self.assertEqual(chart['policy_mode'], 'deterministic')
        for history, actions in validate_chart(chart).items():
            for hand in COMBOS:
                probabilities = [r.get(hand, 0) for r in actions.values()]
                self.assertTrue(all(p in (0, 1) for p in probabilities))
                self.assertEqual(sum(probabilities), 0 if hand_class(hand) in chart['unobserved_hands'][history] else 1)

    def test_preflop_uses_most_likely_action(self):
        chart = screenshot_chart()['nodes']
        self.assertEqual(chart['']['fold']['T4o'], 1)
        self.assertEqual(chart['']['raise:2.5']['Q2o'], 1)
        vs_three = chart['raise:2.5/raise:11.5']
        self.assertEqual(vs_three['raise:26']['AKo'], 1)
        self.assertEqual(vs_three['call']['99'], 1)
        # K5s is 50% raise / 50% call: call wins the documented tie.
        self.assertEqual(chart['raise:2.5']['call']['K5s'], 1)

    def test_postflop_policy_is_one_hot_even_with_ties(self):
        for advantages, expected in [([-2., 1., 3.], [0, 0, 1]),
                                     ([-2., -1.], [0, 1]), ([0., 0.], [1, 0])]:
            np.testing.assert_array_equal(greedy_policy(np.array(advantages)), expected)
