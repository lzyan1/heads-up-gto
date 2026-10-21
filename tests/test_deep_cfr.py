"""Optional neural tests: run with pip install '.[deep]' to exercise them."""
import importlib.util
import json
from dataclasses import replace
import subprocess
import sys
import unittest
from unittest.mock import patch

from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State
from heads_up_gto.backends import create_solver
from heads_up_gto.diagnostics import river_exploitability

HAS_DEEP = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_DEEP:
    import numpy as np
    import torch
    from heads_up_gto.deep_cfr import (DeepCFRSolver, FEATURES, MAX_ACTIONS, Reservoir,
                                       encode, regret_match, weighted_loss)


def root():
    return State(2, (3, 3), parse_cards('2c 7d 9h Js Qc'), street=2)


def ranges():
    return parse_range('AsAh,6s6h'), parse_range('KsKh,5s5h')


class BackendTests(unittest.TestCase):
    def test_dependency_free_default_and_actionable_optional_error(self):
        code = """
from heads_up_gto.backends import create_solver
from heads_up_gto.game import Settings, State
from heads_up_gto.cards import parse_cards, parse_range
r = State(2, (3,3), parse_cards('2c 7d 9h Js Qc'), street=2)
ranges = (parse_range('AsAh'), parse_range('KsKh'))
s = create_solver(r, ranges, Settings())
assert s.train(1)['iterations'] == 1
try:
    create_solver(r, ranges, Settings(mode='exact', backend='deep_cfr'))
except ValueError as exc:
    assert "pip install '.[deep]'" in str(exc)
else:
    raise AssertionError('Optional dependency was not checked')
"""
        subprocess.run([sys.executable, '-S', '-c', code], check=True, capture_output=True)

    def test_settings_reject_unbounded_or_incompatible_neural_configuration(self):
        base = dict(backend='deep_cfr', mode='exact')
        for changes in (dict(deep_buffer_size=100001), dict(deep_hidden=True),
                        dict(deep_device='cuda'), dict(mode='bucketed'),
                        dict(chance_mode='enumerate_turns'), dict(max_nodes=10)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Settings.from_dict({**base, **changes})
        for state in (replace(root(), pending=True), replace(root(), terminal='showdown'),
                      replace(root(), street=0), replace(root(), street=-1)):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'active flop, turn or river'):
                create_solver(state, ranges(), Settings(**base))


@unittest.skipUnless(HAS_DEEP, 'Install .[deep] for neural validation')
class DeepCFRTests(unittest.TestCase):
    def solver(self, root_state=None, **kwargs):
        configuration = dict(backend='deep_cfr', mode='exact', deep_hidden=32,
                             deep_buffer_size=256, deep_traversals=8, deep_train_steps=8,
                             deep_batch_size=16, deep_inference_batch=4)
        return DeepCFRSolver(root_state or root(), ranges(), Settings(**{**configuration, **kwargs}))

    def test_encoder_preserves_visible_information_without_card_order_artifacts(self):
        state, hand = root(), next(iter(ranges()[0]))
        settings = Settings()
        def features(s, h=hand): return encode(s, h, s.actions(settings), 4)
        original = features(state)
        self.assertEqual(original.shape, (FEATURES,))
        np.testing.assert_array_equal(original, features(state, hand[::-1]))
        np.testing.assert_array_equal(original, features(replace(state, board=(*state.board[:3][::-1], *state.board[3:]))))
        for changed in (replace(state, board=(*state.board[:3], *state.board[3:][::-1])),
                        replace(state, path=('check',)), replace(state, paid=(1, 0)),
                        replace(state, aggressor_history=(1, 0))):
            self.assertFalse(np.array_equal(original, features(changed)))
        # Only state, the acting player's hand, actions and a fixed scale are
        # inputs. Unseen turn/river slots remain empty on a turn-root solve.
        turn = replace(state, street=1, board=state.board[:4])
        self.assertEqual(np.count_nonzero(features(turn)[3 * 69:4 * 69]), 0)
        flop = replace(state, street=0, board=state.board[:3])
        self.assertEqual(np.count_nonzero(features(flop)[2 * 69:4 * 69]), 0)

    def test_flop_training_samples_both_later_streets_and_inspects_one_model(self):
        flop = replace(root(), street=0, board=root().board[:3])
        solver = self.solver(root_state=flop, sizes=((.5,), (.5,), (.5,)), max_raises=0)
        # Deals vary beyond the fixed flop; neither inspection runout is locked in.
        deals = [solver.sample_deal()[1] for _ in range(8)]
        self.assertTrue(all(board[:3] == flop.board for board in deals))
        self.assertGreater(len({board[3:] for board in deals}), 1)
        solver.train(2)
        memories = solver.advantage_memory + solver.strategy_memory
        seen_streets = set()
        for memory in memories:
            seen_streets.update(np.argmax(memory.features[:memory.size, 4 * 69 + 2:4 * 69 + 5], axis=1))
        self.assertEqual(seen_streets, {0, 1, 2})
        before = solver.stats()
        for path in ([], ['check'], ['check', 'check', 'deal:Ts'],
                     ['check', 'check', 'deal:Ts', 'check', 'check', 'deal:3d']):
            view = solver.inspect(path)
            self.assertAlmostEqual(view['prediction_coverage'], 1)
            self.assertAlmostEqual(sum(view['aggregate']), 1, places=6)
            self.assertTrue(all(len(r['probabilities']) == len(view['actions']) for r in view['rows']))
        self.assertEqual(solver.stats(), before)

    def test_reservoir_is_bounded_and_samples_the_entire_stream(self):
        retained = np.zeros(100, dtype=int)
        for seed in range(300):
            memory = Reservoir(10, seed)
            allocation = memory.nbytes
            features = np.zeros(FEATURES, dtype=np.float32)
            for i in range(100):
                features[0] = i
                memory.add(features, [i, -i], i + 1)
            self.assertEqual(memory.size, 10)
            self.assertEqual(memory.seen, 100)
            self.assertEqual(memory.nbytes, allocation)
            self.assertTrue(np.all(memory.masks[:, :2]))
            self.assertFalse(np.any(memory.masks[:, 2:]))
            for index in memory.features[:, 0].astype(int):
                retained[index] += 1
        self.assertTrue(np.all(retained > 10))
        self.assertLess(abs(retained[:50].sum() - retained[50:].sum()), 160)

    def test_regret_matching_and_iteration_weighted_masked_loss(self):
        np.testing.assert_allclose(regret_match(np.array([-2., 1., 3.])), [0, .25, .75])
        np.testing.assert_allclose(regret_match(np.array([-2., -1.])), [0, 1])
        np.testing.assert_allclose(regret_match(np.array([0., 0.])), [.5, .5])
        prediction = torch.tensor([[1., 999.], [1., 999.]])
        target = torch.zeros((2, 2))
        mask = torch.tensor([[True, False], [True, False]])
        loss = weighted_loss(prediction, target, mask, torch.tensor([1., 3.]), 3)
        self.assertAlmostEqual(loss.item(), 4 / 3, places=6)
        logits = torch.tensor([[0., 0., 999.]])
        self.assertEqual(weighted_loss(logits, torch.tensor([[.5, .5, 0.]]),
                                       torch.tensor([[True, True, False]]), torch.tensor([1.]), 1, True).item(), 0)

    def test_unseen_predictions_are_separate_from_sampling_and_use_average_model(self):
        solver = self.solver()
        self.assertIsNone(solver.policy(root(), next(iter(ranges()[0])))[0])
        solver.train(1)
        hand = next(iter(parse_range('AcAd')))
        probabilities, updates, averages = solver.policy(root(), hand)
        self.assertAlmostEqual(sum(probabilities), 1, places=6)
        self.assertEqual((updates, averages), (0, 0))
        # Zero all visit counters: predictions still exist; direct coverage is 0.
        solver.nodes.clear()
        view = solver.inspect([])
        self.assertEqual(view['coverage'], 0)
        self.assertAlmostEqual(view['prediction_coverage'], 1)
        self.assertTrue(view['neural']['predictions'])
        self.assertAlmostEqual(sum(view['aggregate']), 1, places=6)
        before = solver.policy(root(), hand)[0]
        with torch.no_grad():
            for parameter in solver.advantages[0].parameters(): parameter.fill_(123)
        solver._policy_cache.clear()
        self.assertEqual(before, solver.policy(root(), hand)[0])
        json.dumps(solver.stats(), allow_nan=False)
        json.dumps(view, allow_nan=False)

    def test_stop_and_resume_preserve_partial_round_exactly(self):
        full, resumed = self.solver(), self.solver()
        full.train(2)
        collect = resumed._collect
        def stopping_collect(player, iteration):
            for index, result in enumerate(collect(player, iteration)):
                if player == 0 and iteration == 1 and index == 2:
                    resumed.cancel.set()
                yield result
        resumed._collect = stopping_collect
        stopped = resumed.train(2)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertTrue(stopped['pending_round'])
        self.assertEqual(stopped['iterations'], 0)
        resumed.cancel.clear()
        resumed.train(2)
        self.assertEqual(full.completed_traversals, resumed.completed_traversals)
        for a, b in zip(full.advantage_memory + full.strategy_memory, resumed.advantage_memory + resumed.strategy_memory):
            self.assertEqual(a.seen, b.seen)
            np.testing.assert_array_equal(a.targets[:a.size], b.targets[:b.size])
        for a, b in zip(full.average_models, resumed.average_models):
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    def test_short_time_budget_and_unavailable_mps(self):
        solver = self.solver()
        result = solver.train(time_limit_seconds=1e-9)
        self.assertEqual(result['iterations'], 0)
        self.assertFalse(result['prediction_ready'])
        with patch('torch.backends.mps.is_available', return_value=False), self.assertRaisesRegex(ValueError, 'MPS is unavailable'):
            self.solver(deep_device='mps')

    def test_pausing_an_optimizer_and_bounded_visit_tracking(self):
        full, resumed = self.solver(), self.solver()
        full.train(1)
        fit = resumed._fit
        def stopping_fit(memory, iteration, player, strategy=False):
            generator = fit(memory, iteration, player, strategy)
            index = 0
            while True:
                try:
                    value = next(generator)
                except StopIteration as done:
                    return done.value
                if player == 1 and strategy and index == 3:
                    resumed.cancel.set()
                index += 1
                yield value
        resumed._fit = stopping_fit
        self.assertEqual(resumed.train(1)['status'], 'cancelled')
        self.assertFalse(resumed.stats()['prediction_ready'])
        resumed.cancel.clear()
        resumed.train(1)
        for a, b in zip(full.average_models, resumed.average_models):
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        limited = self.solver()
        with patch('heads_up_gto.deep_cfr.TRACKING_LIMIT', 1):
            limited.train(1)
        self.assertEqual(len(limited.nodes), 1)
        self.assertTrue(limited.inspect([])['neural']['tracking_saturated'])
        self.assertTrue(limited.stats()['prediction_ready'])

    def test_exact_river_exploitability_improves(self):
        solver = self.solver(deep_hidden=64, deep_buffer_size=4000, deep_traversals=64,
                             deep_train_steps=128, deep_batch_size=128)
        before = river_exploitability(solver)['exploitability_bb']
        solver.train(20)
        after = river_exploitability(solver)['exploitability_bb']
        self.assertLess(after, before * .6)
        self.assertEqual(solver.completed_traversals, [1280, 1280])


if __name__ == '__main__':
    unittest.main()
