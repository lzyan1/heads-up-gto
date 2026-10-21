import json
from dataclasses import replace
import unittest

from heads_up_gto.cards import parse_cards
from heads_up_gto.reference_line import comparison_report, supports_reference
from heads_up_gto.server import prepare


class ReferenceLineTests(unittest.TestCase):
    def solver(self):
        return prepare({'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call',
                        'settings': {'mode': 'exact'}})[0]

    def test_reference_uses_existing_sizes_and_does_not_train(self):
        solver = self.solver()
        before = solver.stats()
        report = comparison_report(solver)
        self.assertEqual(solver.stats(), before)
        self.assertEqual(report['path'], ['check', 'bet:1.7', 'call', 'deal:2h',
                                          'check', 'bet:6.3', 'call', 'deal:3d'])
        self.assertFalse(report['exact_benchmark'])
        self.assertFalse(report['reference_frequencies_available'])
        for entry in report['nodes']:
            model = entry['model']
            state, _, _ = solver.locate(model['path'])
            self.assertEqual(model['actions'], [a.as_dict() for a in state.actions(solver.settings)])
            self.assertIsNone(entry['frequency_error'])
            self.assertIsNone(entry['reference']['probabilities'])
            self.assertEqual(model['aggregate'], [])
        self.assertFalse(report['nodes'][0]['same_menu'])
        self.assertTrue(report['nodes'][0]['same_amounts'])
        self.assertFalse(report['nodes'][-1]['same_amounts'])
        self.assertEqual(report['nodes'][-1]['model']['board'], ['As', 'Kd', '7c', '2h', '3d'])
        self.assertEqual(report['nodes'][-1]['model']['stacks'], (89.5, 89.5))
        json.dumps(report, allow_nan=False)

    def test_only_matching_flop_root_can_use_reference(self):
        solver = self.solver()
        self.assertTrue(supports_reference(solver.root))
        for root in (replace(solver.root, board=parse_cards('As Kd 8c')),
                     replace(solver.root, base_pot=23),
                     replace(solver.root, actor=1, checked=True),
                     replace(solver.root, board=parse_cards('As Kd 7c 2h'), street=1)):
            solver.root = root
            self.assertFalse(supports_reference(root))
            with self.assertRaisesRegex(ValueError, 'original As Kd 7c flop'):
                comparison_report(solver)
