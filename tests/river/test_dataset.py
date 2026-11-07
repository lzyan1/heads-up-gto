"""Offline generator, label semantics, durable output and worker lifecycle."""
from collections import Counter
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest

from heads_up_gto.cards import combo_text, parse_cards, parse_range, remove_blocked
from heads_up_gto.game import Settings, State
from heads_up_gto.river.naive_bots import NaiveBot, hand_features
from heads_up_gto.river.dataset import (DatasetStore, FORMATIONS, SolveWorker, SpotGenerator,
                                       attempt_seed, next_formation, solve_spot)


def kuhn_spot():
    root = State(2, (1, 1), parse_cards('2c 5d 7h 9s Jc'), street=2)
    hands = parse_range('As3s,Kh3h,Qd3d')
    # Generic Settings defaults include extra sizes; at stack 1 all clamp to the
    # same one-chip bet. Disable raises via effective stacks, as in real Kuhn.
    return dict(state=root, ranges=(hands, hands), history=[], bots=[], sampled_hands=[])


class BotTests(unittest.TestCase):
    def feature(self, hand, board):
        return hand_features(tuple(sorted(parse_cards(hand))), parse_cards(board))

    def test_hand_categories_and_draws_have_expected_order(self):
        examples = [('7s3d', 'AhKd2c', 'air'), ('7s7d', 'AhKd2c', 'weak_pair'),
                    ('KsQd', 'AhKd2c', 'middle_pair'), ('AsQd', 'AhKd2c', 'top_pair'),
                    ('AsAd', 'Kh9d2c', 'overpair'), ('AsKd', 'AhKd2c', 'two_pair'),
                    ('AsAd', 'AhKd2c', 'trips'), ('As5d', '2c3d4h', 'straight'),
                    ('AsQs', '2s7s9s', 'flush')]
        previous = -1
        for hand, board, label in examples:
            value, _, actual = self.feature(hand, board)
            self.assertEqual(actual, label)
            self.assertGreater(value, previous)
            previous = value
        self.assertEqual(self.feature('AsKh', 'QhJh2c')[2], 'gutshot')
        self.assertEqual(self.feature('9s8d', '7h6c2d')[2], 'open_ended')
        self.assertEqual(self.feature('AsQs', '2s7s9h')[2], 'flush_draw')
        self.assertEqual(self.feature('9s8s', '7s6c2s')[2], 'combo_draw')
        self.assertEqual(self.feature('AsKd', '2h2c7d')[2], 'overcards')

    def test_policy_is_deterministic_and_uses_only_visible_state(self):
        state = State(5, (97.5, 97.5), parse_cards('AsKd7c'), sizing_profile='single_raised')
        actions = state.actions(Settings())
        bot = NaiveBot(7, 3)
        for text in ('AhQd', '7s7d', 'Qh9d'):
            hand = next(iter(parse_range(text)))
            self.assertEqual(bot.action(state, hand, actions), bot.action(state, hand, actions))
            self.assertIn(bot.action(state, hand, actions), actions)
        self.assertEqual(NaiveBot.sample(random.Random(7)), NaiveBot.sample(random.Random(7)))


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = SpotGenerator()

    def test_formation_mix_counts_successful_samples(self):
        counts = Counter()
        for _ in range(100):
            counts[next_formation(counts)] += 1
        self.assertEqual(dict(counts), dict(single_raised=45, three_bet=45, four_bet=10))
        self.assertEqual(attempt_seed(7, 99), attempt_seed(7, 99))
        self.assertNotEqual(attempt_seed(7, 99), attempt_seed(7, 100))

    def test_rollouts_reproduce_and_ranges_match_every_public_action(self):
        for name in FORMATIONS:
            found = None
            for attempt in range(50):
                seed = attempt_seed(7, attempt)
                spot, outcome = self.generator.play(name, seed)
                if outcome == 'river':
                    found = spot
                    self.assertEqual(self.generator.play(name, seed), (spot, outcome))
                    break
            self.assertIsNotNone(found, name)
            root, initial, _, _ = self.generator.formations[name]
            state = State(root.pot, root.stacks, spot['state'].board[:3], sizing_profile=name)
            ranges = tuple(remove_blocked(r, state.board) for r in initial)
            bots = [NaiveBot(**b) for b in spot['bots']]
            for token in spot['history']:
                if token.startswith('deal:'):
                    state = state.deal(parse_cards(token[5:])[0])
                    ranges = tuple(remove_blocked(r, state.board) for r in ranges)
                else:
                    actions = state.actions(Settings())
                    actor = state.actor
                    # Independently rebuild the complete conditioned range.
                    retained = {}
                    for hand, weight in ranges[actor].items():
                        if bots[actor].action(state, hand, actions).key == token:
                            retained[hand] = weight
                    ranges = tuple(retained if p == actor else r for p, r in enumerate(ranges))
                    state = state.apply(next(a for a in actions if a.key == token))
            self.assertEqual(state, spot['state'])
            self.assertEqual(ranges, spot['ranges'])
            self.assertEqual(len(state.board), 5)
            self.assertEqual(state.street, 2)
            self.assertFalse(state.terminal or state.pending)
            self.assertGreater(min(state.stacks), 0)
            self.assertEqual(state.pot + sum(state.stacks), 200)
            hands = [next(iter(parse_range(h))) for h in spot['sampled_hands']]
            self.assertFalse(set(hands[0]).intersection(hands[1]))
            for player, hand in enumerate(hands):
                self.assertIn(hand, ranges[player])
                self.assertFalse(set(hand).intersection(state.board))
            for r in ranges:
                self.assertTrue(all(w == 1 for w in r.values()))
                self.assertTrue(all(not set(h).intersection(state.board) for h in r))


class LabelTests(unittest.TestCase):
    def test_solved_labels_match_analytic_kuhn_values(self):
        sample = solve_spot(kuhn_spot(), 10, 1e-9, 25000000)
        for text, expected in [('As3s', 7/6), ('Kh3h', -1/3), ('Qd3d', -1)]:
            key = combo_text(next(iter(parse_range(text))))
            self.assertAlmostEqual(sample['hand_ev_bb'][0][key], expected)
            self.assertAlmostEqual(sample['hand_ev_pot'][0][key], expected / 2)
        self.assertLess(sample['certificate']['exploitability_bb'], 1e-9)
        self.assertEqual(sample['board'], ['2c', '5d', '7h', '9s', 'Jc'])
        json.dumps(sample, allow_nan=False)

    def test_incompatible_private_hands_are_masked_not_given_zero_ev(self):
        spot = kuhn_spot()
        spot['ranges'] = (parse_range('As3s,Kh3h'), parse_range('As3s'))
        sample = solve_spot(spot, 10, 1e-9, 25000000)
        key = combo_text(next(iter(parse_range('As3s'))))
        self.assertIsNone(sample['hand_ev_bb'][0][key])
        self.assertIsNone(sample['hand_ev_pot'][0][key])


class StorageTests(unittest.TestCase):
    def record(self, sample_id, attempt_id=None):
        attempt_id = sample_id if attempt_id is None else attempt_id
        sample = dict(sample_id=sample_id, attempt_id=attempt_id, seed=7,
                      formation='single_raised', board=['As', 'Kd', '7c', '2h', '3d'],
                      ranges=[{'AhAd': 1}, {'KhKc': 1}])
        return dict(status='solved', attempt_id=attempt_id, formation='single_raised', sample=sample)

    def rows(self, folder):
        rows = []
        for path in sorted(Path(folder).glob('spots-*.csv')):
            with path.open() as handle:
                rows.extend(csv.DictReader(handle))
        return rows

    def test_resume_csv_shards_journal_tail_recovery_and_failure_events(self):
        with tempfile.TemporaryDirectory() as folder:
            with DatasetStore(folder, {'seed': 7}, chunk_size=2) as store:
                store.commit(self.record(0))
                store.commit(dict(status='failed', attempt_id=1, formation='three_bet', error='timeout'))
                store.commit(self.record(1, 2))
                store.commit(self.record(2, 3))
            self.assertEqual([r['sample_id'] for r in self.rows(folder)], ['0', '1', '2'])
            with (Path(folder) / 'events.jsonl').open('ab') as handle:
                handle.write(b'{"incomplete":')
            with DatasetStore(folder, {'seed': 7}, chunk_size=2) as store:
                self.assertEqual(store.completed, 3)
                self.assertEqual(store.next_attempt, 4)
                self.assertEqual(store.failures['three_bet'], 1)
                store.commit(self.record(3, 4))
                store.commit(self.record(4, 5))
            rows = self.rows(folder)
            self.assertEqual([r['sample_id'] for r in rows], list(map(str, range(5))))
            self.assertEqual(json.loads(rows[0]['ranges']), [{'AhAd': 1}, {'KhKc': 1}])
            self.assertEqual(len(list(Path(folder).glob('spots-*.csv'))), 3)
            self.assertFalse(list(Path(folder).glob('*.part')))
            events = [json.loads(line) for line in (Path(folder) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 6)

    def test_config_mismatch_and_simultaneous_writers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with DatasetStore(folder, {'seed': 7}):
                with self.assertRaises(BlockingIOError):
                    DatasetStore(folder, {'seed': 7})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                DatasetStore(folder, {'seed': 8})
            self.assertEqual(json.loads((Path(folder) / 'manifest.json').read_text()), {'seed': 7})

    def test_complete_corrupt_event_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as folder:
            with DatasetStore(folder, {'seed': 7}):
                pass
            journal = Path(folder) / 'events.jsonl'
            journal.write_bytes(b'not json\n')
            with self.assertRaises(ValueError):
                DatasetStore(folder, {'seed': 7})
            self.assertEqual(journal.read_bytes(), b'not json\n')


class WorkerTests(unittest.TestCase):
    def test_worker_reuse_timeout_and_cancellation(self):
        payload = kuhn_spot(), 10, 1e-8, 25000000
        stop = threading.Event()
        with SolveWorker() as worker:
            first = worker.solve(payload, 10, stop, time.monotonic() + 20)
            pid = worker.process.pid
            second = worker.solve(payload, 10, stop, time.monotonic() + 20)
            self.assertEqual(worker.process.pid, pid)
            self.assertAlmostEqual(first['certificate']['value_bb'], second['certificate']['value_bb'])
            worker.close()
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                worker.solve(payload, .0001, stop, time.monotonic() + 20)
            self.assertIsNone(worker.process)
            stop.set()
            with self.assertRaises(InterruptedError):
                worker.solve(payload, 10, stop, time.monotonic() + 20)
            self.assertIsNone(worker.process)


if __name__ == '__main__':
    unittest.main()
