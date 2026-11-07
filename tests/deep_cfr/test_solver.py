"""Deep CFR learning, representation, bounded memory and resume validation."""
import json
from dataclasses import replace
import unittest
from unittest.mock import patch

from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State
from heads_up_gto.diagnostics import river_exploitability

import numpy as np
import torch
from heads_up_gto.deep_cfr.solver import (DeepCFRSolver, FEATURES, MAX_ACTIONS, Reservoir,
                                   encode, greedy_policy, weighted_loss)


def root():
    return State(2, (3, 3), parse_cards('2c 7d 9h Js Qc'), street=2)


def ranges():
    return parse_range('AsAh,6s6h'), parse_range('KsKh,5s5h')


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_removed_solver_options(self):
        self.assertEqual(Settings.from_dict({}).deep_device, 'cpu')
        for name, value in dict(backend='tabular', mode='exact', max_nodes=1, buckets=8).items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unknown settings'):
                Settings.from_dict({name: value})

    def test_invalid_neural_or_river_configuration(self):
        for changes in (dict(deep_buffer_size=100001), dict(deep_hidden=True), dict(deep_device='cuda'),
                        dict(seed=True), dict(river_seconds=0), dict(river_tolerance=float('nan'))):
            with self.assertRaises(ValueError):
                Settings.from_dict(changes)
        with self.assertRaisesRegex(ValueError, 'direct solving'):
            DeepCFRSolver(root(), ranges(), Settings())


class ImmediateRiverJob:
    # Real LP values, without process startup in the repeated neural unit tests.
    def __init__(self, root, ranges, settings, **kwargs):
        self.root, self.ranges, self.settings = root, ranges, settings
    def poll(self):
        from heads_up_gto.river.exact_lp import ExactRiver
        return ExactRiver(self.root, self.ranges, self.settings).solve().compact()
    def pause(self): pass


class DeepCFRTests(unittest.TestCase):
    def setUp(self):
        self.worker_patch = patch('heads_up_gto.deep_cfr.solver.RiverJob', ImmediateRiverJob)
        self.worker_patch.start()
    def tearDown(self):
        self.worker_patch.stop()
    def solver(self, root_state=None, **kwargs):
        configuration = dict(deep_hidden=32, deep_buffer_size=256, deep_traversals=4,
                             deep_train_steps=8, deep_batch_size=16, deep_inference_batch=2)
        turn = replace(root(), board=root().board[:4], street=1)
        return DeepCFRSolver(root_state or turn, ranges(), Settings(**{**configuration, **kwargs}))

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
        np.testing.assert_allclose(greedy_policy(np.array([-2., 1., 3.])), [0, 0, 1])
        np.testing.assert_allclose(greedy_policy(np.array([-2., -1.])), [0, 1])
        np.testing.assert_allclose(greedy_policy(np.array([0., 0.])), [1, 0])
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
        self.assertIsNone(solver.policy(solver.root, next(iter(ranges()[0])))[0])
        solver.train(1)
        hand = next(iter(parse_range('AcAd')))
        probabilities, updates, averages = solver.policy(solver.root, hand)
        self.assertAlmostEqual(sum(probabilities), 1, places=6)
        self.assertEqual((updates, averages), (0, 0))
        # Zero all visit counters: predictions still exist; direct coverage is 0.
        solver.nodes.clear()
        view = solver.inspect([])
        self.assertEqual(view['coverage'], 0)
        self.assertAlmostEqual(view['prediction_coverage'], 1)
        self.assertTrue(view['neural']['predictions'])
        self.assertAlmostEqual(sum(view['aggregate']), 1, places=6)
        before = solver.policy(solver.root, hand)[0]
        with torch.no_grad():
            for parameter in solver.advantages[0].parameters(): parameter.fill_(123)
        solver._policy_cache.clear()
        self.assertEqual(before, solver.policy(solver.root, hand)[0])
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
        with patch('heads_up_gto.deep_cfr.solver.TRACKING_LIMIT', 1):
            limited.train(1)
        self.assertEqual(len(limited.nodes), 1)
        self.assertTrue(limited.inspect([])['neural']['tracking_saturated'])
        self.assertTrue(limited.stats()['prediction_ready'])


    def test_timed_pauses_get_fresh_budgets_and_resume_the_same_work(self):
        full, resumed = self.solver(), self.solver()
        full.train(1)
        clock = [100.0]
        collect = resumed._collect
        def timed_collect(player, iteration):
            for result in collect(player, iteration):
                clock[0] += .3
                yield result
        resumed._collect = timed_collect
        with patch('heads_up_gto.deep_cfr.solver.time.monotonic', side_effect=lambda: clock[0]):
            first = resumed.train(time_limit_seconds=.5)
            second = resumed.train(time_limit_seconds=.5)
        self.assertEqual((first['run_seconds'], second['run_seconds']), (.6, .6))
        self.assertEqual(second['seconds'], 1.2)
        self.assertTrue(second['pending_round'])
        resumed._collect = collect
        resumed.train(1)
        self.assertEqual(full.completed_traversals, resumed.completed_traversals)
        for a, b in zip(full.average_models, resumed.average_models):
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)


    def test_budget_validation_and_manual_stop(self):
        solver = self.solver()
        for kwargs in ({}, {'iterations': 1, 'time_limit_seconds': 1}, {'iterations': 0},
                       {'time_limit_seconds': 0}, {'time_limit_seconds': -1},
                       {'time_limit_seconds': float('nan')}, {'time_limit_seconds': float('inf')},
                       {'time_limit_seconds': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.train(**kwargs)
        solver.cancel.set()
        self.assertEqual(solver.train(time_limit_seconds=10)['status'], 'cancelled')
        self.assertEqual(solver.iterations, 0)


    def test_failed_round_retains_published_policy(self):
        solver = self.solver()
        solver.train(1)
        before = solver.inspect([])
        with patch.object(solver, '_deal', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                solver.train(1)
        self.assertEqual(solver.stats()['status'], 'error')
        self.assertEqual(solver.inspect([]), before)
        with self.assertRaisesRegex(ValueError, 'new solve or refinement'):
            solver.train(1)


    def test_blocked_selected_hand_and_illegal_path(self):
        solver = self.solver()
        self.assertIn('selected_error', solver.inspect([], parse_cards('2c As')))
        with self.assertRaises(ValueError):
            solver.inspect(['bet:9999'])


    def test_river_is_a_direct_boundary_and_has_no_neural_training_samples(self):
        solver = self.solver()
        solver.train(2)
        self.assertGreater(solver.river_solves, 0)
        self.assertLess(solver.last_river_certificate['exploitability_bb'], 1e-7)
        for memory in solver.advantage_memory + solver.strategy_memory:
            streets = np.argmax(memory.features[:memory.size, 4 * 69 + 2:4 * 69 + 5], axis=1)
            self.assertTrue(np.all(streets == 1))
        view = solver.inspect([])
        self.assertTrue(all(set(r['probabilities']) <= {0., 1.} for r in view['rows']))

    def test_river_range_propagation_uses_current_deterministic_policy(self):
        solver = self.solver()
        state = solver.root
        for key in ('check', 'check'):
            state = state.apply(next(a for a in state.actions(solver.settings) if a.key == key))
        state = state.deal(parse_cards('Qc')[0])
        conditioned, off_path = solver._river_ranges(state)
        self.assertFalse(off_path)
        self.assertEqual(conditioned, solver.ranges)
        solver._training_policy = lambda state, hands: dict.fromkeys(hands, 1)
        _, off_path = solver._river_ranges(state)
        self.assertTrue(off_path)

    def test_river_inspection_reuses_the_boundary_policy_for_later_actions(self):
        solver = self.solver()
        path = ['check', 'check', 'deal:Qc']
        root_view = solver.inspect(path)
        self.assertIn('exact', root_view)
        self.assertEqual(len(solver._inspection_rivers), 1)
        solver.inspect(path + ['check'])
        self.assertEqual(len(solver._inspection_rivers), 1)
        self.assertTrue(all(r['policy_source'] == 'exact_river' for r in root_view['rows']))


if __name__ == '__main__':
    unittest.main()
