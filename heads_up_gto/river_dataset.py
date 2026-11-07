"""Offline naive-bot river generation and verified LP labels. No neural training."""
import argparse
from collections import Counter
from contextlib import AbstractContextManager
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import threading
import time

from .cards import DECK, card_text, combo_text, remove_blocked
from .game import Settings, State
from .naive_bots import BOT_VERSION, NaiveBot
from .preflop import chart_ranges, reconstruct
from .range_game import Distribution
from .screenshot_charts import screenshot_chart
from .sizing import MODEL_VERSION

VERSION = 1
FORMATIONS = {
    'single_raised': ('raise:2.5/call', 45),
    'three_bet': ('raise:2.5/raise:11.5/call', 45),
    'four_bet': ('raise:2.5/raise:11.5/raise:26/call', 10),
}
FIELDS = ('sample_id', 'attempt_id', 'seed', 'formation', 'board', 'pot_bb',
          'stacks_bb', 'actor', 'history', 'root_state', 'ranges', 'bots',
          'sampled_hands', 'hand_ev_bb', 'hand_ev_pot', 'root_actions',
          'root_strategy', 'certificate', 'label_seconds')


def encoded(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False)


def attempt_seed(seed, attempt):
    return int.from_bytes(hashlib.sha256(f'{seed}:{attempt}'.encode()).digest()[:8], 'big')


class SpotGenerator:
    def __init__(self):
        chart = screenshot_chart()
        self.formations = {}
        for name, (line, _) in FORMATIONS.items():
            preflop = reconstruct(line)
            ranges = chart_ranges(chart, preflop, ())
            mass = sum(ranges[1].values())
            card_mass = [0.] * 52
            for hand, w in ranges[1].items():
                for card in hand:
                    card_mass[card] += w
            first = {h: w * (mass - card_mass[h[0]] - card_mass[h[1]] + ranges[1].get(h, 0))
                     for h, w in ranges[0].items()}
            first = {h: w for h, w in first.items() if w > 1e-10}
            self.formations[name] = preflop, ranges, Distribution(first), Distribution(ranges[1])

    def play(self, formation, seed):
        """One honest rollout. Return (river inputs, outcome); folds are not forced through."""
        rng = random.Random(seed)
        preflop, initial, first, second = self.formations[formation]
        h0 = first.draw(rng)
        for _ in range(40):
            h1 = second.draw(rng)
            if not set(h0).intersection(h1):
                break
        else:
            h1 = Distribution(remove_blocked(initial[1], h0)).draw(rng)
        hands = h0, h1
        blocked = set((*h0, *h1))
        board = rng.sample([c for c in DECK if c not in blocked], 5)
        state = State(preflop.pot, preflop.stacks, tuple(board[:3]), sizing_profile=formation)
        ranges = tuple(remove_blocked(r, state.board) for r in initial)
        bots = NaiveBot.sample(rng), NaiveBot.sample(rng)
        settings, history = Settings(), []
        while not state.terminal:
            if state.pending:
                card = board[len(state.board)]
                state = state.deal(card)
                history.append('deal:' + card_text(card))
                ranges = tuple(remove_blocked(r, state.board) for r in ranges)
            if state.street == 2:
                # Both sampled hands remain in these ranges; never expose them
                # to the LP or condition ranges on the opponent's actual hand.
                return dict(state=state, ranges=ranges, bots=[asdict(b) for b in bots],
                            sampled_hands=[combo_text(h) for h in hands], history=history), 'river'
            actions = state.actions(settings)
            actor = state.actor
            action = bots[actor].action(state, hands[actor], actions)
            updated = {h: w for h, w in ranges[actor].items()
                       if bots[actor].action(state, h, actions).key == action.key}
            if hands[actor] not in updated:
                raise RuntimeError('Range filtering removed the hand that took the action.')
            ranges = tuple(updated if p == actor else r for p, r in enumerate(ranges))
            history.append(action.key)
            state = state.apply(action)
        return None, 'fold' if state.terminal == 'fold' else 'all_in'


def solve_spot(spot, lp_seconds, tolerance, max_entries):
    """Per-hand conditional on-policy EVs under the certified river equilibrium.

    Positive range hands with no compatible opponent get null labels, not zero
    EV. Unrepresented/zero-weight hands have no labels in this dataset version.
    """
    import numpy as np
    from .exact_river import ExactRiver
    start = time.monotonic()
    result = ExactRiver(spot['state'], spot['ranges'], Settings(), max_payoff_entries=max_entries)
    result.solve(time_limit_seconds=lp_seconds, tolerance=tolerance)
    values, _ = result.certify()
    evs = []
    for player, indices in enumerate((result.pair_i, result.pair_j)):
        mass = np.bincount(indices, weights=result.chance, minlength=len(result.hands[player]))
        total = np.bincount(indices, weights=result.chance * values * (1 if player == 0 else -1),
                            minlength=len(result.hands[player]))
        evs.append({combo_text(h): float(total[i] / mass[i]) if mass[i] > 0 else None
                    for i, h in enumerate(result.hands[player])})
    root = spot['state']
    return dict(board=[card_text(c) for c in root.board], pot_bb=root.pot,
                stacks_bb=root.stacks, actor=root.actor, history=spot['history'],
                root_state=asdict(root), ranges=[{combo_text(h): w for h, w in r.items()} for r in spot['ranges']],
                bots=spot['bots'], sampled_hands=spot['sampled_hands'], hand_ev_bb=evs,
                hand_ev_pot=[{h: v / root.pot if v is not None else None for h, v in ev.items()} for ev in evs],
                root_actions=[a.as_dict() for a in root.actions(Settings())],
                root_strategy={entry['hand']: entry['probabilities'] for entry in result.root_policy()},
                certificate=result.stats, label_seconds=time.monotonic() - start)


def _worker_loop(connection):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            try:
                payload = connection.recv()
            except EOFError:
                return
            try:
                connection.send(('ok', solve_spot(*payload)))
            except Exception as exc:
                connection.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


class SolveWorker(AbstractContextManager):
    """One reusable process, with a wall timeout covering build + LP + labels."""
    def __init__(self):
        self.process = self.connection = None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.connection.close()
            self.process = self.connection = None

    def __exit__(self, *_):
        self.close()

    def solve(self, payload, timeout, stop, deadline):
        if stop.is_set() or time.monotonic() >= deadline:
            raise InterruptedError('Run stopped before submitting a solve.')
        if self.process is not None and not self.process.is_alive():
            self.close()
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_worker_loop, args=(child,), daemon=True)
            self.process.start()
            child.close()
        try:
            self.connection.send(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.close()
            raise RuntimeError('River worker exited before accepting a spot.') from exc
        limit = time.monotonic() + timeout
        while True:
            if stop.is_set() or time.monotonic() >= deadline:
                self.close()
                raise InterruptedError('Run stopped; the unfinished attempt will be retried on resume.')
            if self.connection.poll(.1):
                try:
                    status, value = self.connection.recv()
                except (EOFError, OSError) as exc:
                    self.close()
                    raise RuntimeError('River worker exited before returning a result.') from exc
                if status != 'ok':
                    raise RuntimeError(value)
                return value
            if not self.process.is_alive() or time.monotonic() >= limit:
                self.close()
                raise RuntimeError('River worker exited or exceeded the per-spot wall timeout.')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        handle.write(encoded(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class DatasetStore(AbstractContextManager):
    """Fsynced JSONL journal is authoritative; CSV shards are rebuildable exports."""
    def __init__(self, output, config, chunk_size=2500):
        if chunk_size < 1:
            raise ValueError('CSV chunk size must be positive.')
        self.output, self.chunk_size = Path(output), chunk_size
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / '.lock').open('a+')
        self.journal = self.csv_file = None
        self.counts, self.failures = Counter(), Counter()
        self.completed = self.next_attempt = 0
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.output / 'manifest.json'
            if manifest.exists():
                if json.loads(manifest.read_text()) != json.loads(encoded(config)):
                    raise ValueError('Dataset configuration/code changed. Use a new output directory or the original settings.')
            else:
                if (self.output / 'events.jsonl').exists():
                    raise ValueError('Journal exists without its manifest; refusing to mix datasets.')
                atomic_json(manifest, config)
            self.path = self.output / 'events.jsonl'
            self.journal = self.path.open('a+b')
            # A killed write may leave an incomplete last line. Keep every
            # complete event; corruption in a complete line is an error.
            self.journal.seek(0)
            while True:
                offset = self.journal.tell()
                line = self.journal.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    self.journal.truncate(offset)
                    self.journal.flush()
                    os.fsync(self.journal.fileno())
                    break
                event = json.loads(line)
                self.next_attempt = max(self.next_attempt, event['attempt_id'] + 1)
                if event['status'] == 'solved':
                    sample = event['sample']
                    if sample['sample_id'] != self.completed:
                        raise ValueError('Dataset sample IDs are not consecutive.')
                    self._csv_append(sample)
                    self.completed += 1
                    self.counts[sample['formation']] += 1
                else:
                    self.failures[event['formation']] += 1
            self.journal.seek(0, os.SEEK_END)
            progress = self.output / 'progress.json'
            if progress.exists():
                self.next_attempt = max(self.next_attempt, json.loads(progress.read_text())['next_attempt'])
        except BaseException:
            # Do not publish a partial recovery or overwrite progress when the
            # manifest/journal is invalid or another writer owns the directory.
            if self.csv_file is not None:
                self.csv_file.close()
            if self.journal is not None:
                self.journal.close()
            self.lock.close()
            raise

    def _csv_append(self, sample):
        if self.csv_file is None:
            number = self.completed // self.chunk_size + 1
            self.csv_path = self.output / f'spots-{number:06d}.csv'
            self.csv_temp = self.csv_path.with_suffix('.csv.part')
            self.csv_file = self.csv_temp.open('w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
            self.csv_writer.writeheader()
        self.csv_writer.writerow({k: encoded(v) if isinstance(v, (list, tuple, dict)) else v
                                  for k, v in sample.items()})
        self.csv_file.flush()
        if (self.completed + 1) % self.chunk_size == 0:
            self._seal_csv()

    def _seal_csv(self):
        if self.csv_file is not None:
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            self.csv_file.close()
            self.csv_file = None
            self.csv_temp.replace(self.csv_path)

    def commit(self, event):
        self.journal.write((encoded(event) + '\n').encode())
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.next_attempt = event['attempt_id'] + 1
        if event['status'] == 'solved':
            self._csv_append(event['sample'])
            self.completed += 1
            self.counts[event['formation']] += 1
        else:
            self.failures[event['formation']] += 1

    def checkpoint(self):
        atomic_json(self.output / 'progress.json', dict(completed=self.completed,
                    next_attempt=self.next_attempt, formations=dict(self.counts), failures=dict(self.failures)))

    def __exit__(self, *_):
        try:
            if self.journal is not None:
                self._seal_csv()
                self.checkpoint()
                self.journal.close()
        finally:
            self.lock.close()


def next_formation(counts, rng=None):
    # Every completed block of 20 has 9/9/2 spots. Randomize the remaining
    # slots on each attempt so a slow formation does not monopolize the run.
    block = sum(counts.values()) // 20 + 1
    remaining = {k: block * (v[1] // 5) - counts[k] for k, v in FORMATIONS.items()}
    eligible = [k for k, n in remaining.items() if n > 0]
    rng = rng or random.Random(sum(counts.values()))
    return rng.choices(eligible, weights=[remaining[k] for k in eligible])[0]


def dataset_config(seed, chunk_size, lp_seconds, wall_seconds, tolerance, max_entries):
    root = Path(__file__).parent
    sources = ('river_dataset.py', 'naive_bots.py', 'cards.py', 'preflop.py',
               'game.py', 'sizing.py', 'screenshot_charts.py', 'range_game.py', 'exact_river.py')
    return dict(version=VERSION, seed=seed, chunk_size=chunk_size,
                formations={k: dict(line=v[0], weight=v[1]) for k, v in FORMATIONS.items()},
                bots=BOT_VERSION, sizing=MODEL_VERSION, lp_seconds=lp_seconds,
                wall_seconds=wall_seconds, tolerance=tolerance, max_payoff_entries=max_entries,
                source_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
                chart_sha256=hashlib.sha256(encoded(screenshot_chart()).encode()).hexdigest(),
                labels='Conditional on-policy equilibrium EV, BB and pot-normalized; null means no compatible opponent. '
                       'Only hands in the input ranges are labeled. Player order: BB/OOP, SB/IP. '
                       'Bots and sampled private hands are provenance, not value-network inputs.')


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be positive and finite.')
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer.')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/river-dataset'))
    parser.add_argument('--hours', type=positive_number, default=8, help='Wall time for this invocation (default: 8).')
    parser.add_argument('--samples', type=positive_int, help='Stop at this TOTAL saved sample count, including previous runs.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--chunk-size', type=positive_int, default=2500)
    parser.add_argument('--lp-seconds', type=positive_number, default=120)
    parser.add_argument('--spot-seconds', type=positive_number, default=180, help='Hard wall timeout including LP setup and labels.')
    parser.add_argument('--tolerance', type=positive_number, default=1e-7)
    parser.add_argument('--max-payoff-entries', type=positive_int, default=25_000_000)
    args = parser.parse_args(argv)
    stop = threading.Event()
    def stop_handler(*_):
        stop.set()
    previous = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    config = dataset_config(args.seed, args.chunk_size, args.lp_seconds, args.spot_seconds,
                            args.tolerance, args.max_payoff_entries)
    deadline = time.monotonic() + args.hours * 3600
    started, report_at = time.monotonic(), 0.
    outcomes = Counter()
    try:
        with DatasetStore(args.output, config, args.chunk_size) as store, SolveWorker() as worker:
            generator = SpotGenerator()
            initial = store.completed
            print(f'45/45/10 river dataset: {store.completed} saved; output {args.output.resolve()}', flush=True)
            while not stop.is_set() and time.monotonic() < deadline and (args.samples is None or store.completed < args.samples):
                attempt = store.next_attempt
                seed = attempt_seed(args.seed, attempt)
                formation = next_formation(store.counts, random.Random(seed ^ 0xF04A710))
                spot, outcome = generator.play(formation, seed)
                outcomes[f'{formation}:{outcome}'] += 1
                if spot is None:
                    store.next_attempt += 1
                else:
                    try:
                        sample = worker.solve((spot, args.lp_seconds, args.tolerance, args.max_payoff_entries),
                                              args.spot_seconds, stop, deadline)
                    except InterruptedError:
                        break
                    except RuntimeError as exc:
                        store.commit(dict(status='failed', attempt_id=attempt, seed=seed,
                                          formation=formation, error=str(exc)))
                        print(f'Attempt {attempt} ({formation}) failed: {exc}', flush=True)
                    else:
                        sample.update(sample_id=store.completed, attempt_id=attempt, seed=seed, formation=formation)
                        store.commit(dict(status='solved', attempt_id=attempt, formation=formation, sample=sample))
                        print(f'Saved {store.completed}: {formation}, {tuple(len(r) for r in spot["ranges"])} hands, '
                              f'{sample["label_seconds"]:.2f}s, gap={sample["certificate"]["exploitability_bb"]:.2g} BB', flush=True)
                if time.monotonic() >= report_at:
                    store.checkpoint()
                    elapsed = time.monotonic() - started
                    print(f'Progress: {store.completed} total ({store.completed - initial} new), '
                          f'formations={dict(store.counts)}, attempts this run={sum(outcomes.values())}, '
                          f'outcomes={dict(outcomes)}, elapsed={elapsed / 60:.1f}m', flush=True)
                    report_at = time.monotonic() + 30
            print(f'Finished: {store.completed} saved, formations={dict(store.counts)}. '
                  'Run the same command to resume.', flush=True)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
