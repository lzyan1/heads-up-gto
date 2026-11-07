"""Exact river LP latency on random Hold'em ranges and a Kuhn-in-Hold'em check.

Install the project dependencies, then run from the repository with PYTHONPATH=.
Each case runs in a fresh process to isolate peak memory. Timings separately
record game construction, both equilibrium LPs, and recursive best responses.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

from heads_up_gto.cards import COMBOS, combo_text, evaluate, parse_cards, parse_range, remove_blocked
from heads_up_gto.game import Settings, State
from heads_up_gto.river.exact_lp import ExactRiver

SPOTS = {
    'single_raised_dry': dict(board='As Kd 7c 2h 3d', pot=21, stack=89.5, profile='single_raised', history=(1, 1)),
    'three_bet_flush': dict(board='9h 8h 6c 2h Qd', pot=60, stack=70, profile='three_bet', history=(0, 0)),
    'four_bet_paired': dict(board='Ks Kd 7c 7h 2d', pot=100, stack=50, profile='four_bet', history=(1, 1)),
}


def kuhn():
    root = State(2, (1, 1), parse_cards('2c 5d 7h 9s Jc'), street=2)
    ranges = (parse_range('As3s,Kh3h,Qd3d'),) * 2
    settings = Settings(sizes=((), (), (.5,)), max_raises=0)
    solver = ExactRiver(root, ranges, settings).solve(tolerance=1e-10)
    error = abs(solver.stats['value_bb'] + 1 / 18)
    assert error < 1e-10 and solver.stats['legal_pairs'] == 6
    strategies = []
    for path, node in solver.nodes.items():
        for h in solver.hands[node.state.actor]:
            strategies.append(dict(path=path, actor=node.state.actor, hand=combo_text(h),
                                   actions=[a.key for a in node.actions], probabilities=solver.policy(node.state, h)[0]))
    return dict(board=[*root.board], board_text='2c 5d 7h 9s Jc',
                ranges=['As3s,Kh3h,Qd3d'] * 2, pot=2, stack=1,
                expected_value_bb=-1 / 18, absolute_value_error_bb=error,
                stats=solver.stats, policies=strategies)


def run_case(spot, count, seed, repeats, lp_seconds):
    import scipy
    spec = SPOTS[spot]
    root = State(spec['pot'], (spec['stack'],) * 2, parse_cards(spec['board']), street=2,
                 sizing_profile=spec['profile'], aggressor_history=spec['history'])
    available = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
    if count == 'full':
        n = len(available)
    else:
        n = int(count)
    rng = random.Random(seed)
    ranges = [{h: rng.uniform(.05, 1.) for h in rng.sample(available, n)} for _ in (0, 1)]
    result = dict(spot=spot, board=spec['board'], pot=root.pot, stacks=root.stacks,
                  hands_per_player=n, seed=seed, lp_budget_seconds=lp_seconds,
                  settings=asdict(Settings()), scipy_version=scipy.__version__,
                  ranges=[{combo_text(h): w for h, w in r.items()} for r in ranges], runs=[])
    for _ in range(repeats):
        evaluate.cache_clear()
        start = time.perf_counter()
        solver = None
        try:
            solver = ExactRiver(root, ranges, Settings())
            solver.solve(time_limit_seconds=lp_seconds, tolerance=1e-7)
        except (RuntimeError, ValueError, MemoryError) as exc:
            record = dict(status='failed', error=str(exc), elapsed_seconds=time.perf_counter() - start,
                          partial_stats=solver.stats if solver else None)
        else:
            record = dict(status='certified', **solver.stats)
        record['process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
        result['runs'].append(record)
        if record['status'] != 'certified':
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[3, 10, 30, 100, 200])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 23])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--lp-seconds', type=float, default=20)
    parser.add_argument('--include-full', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/exact-river.json'))
    parser.add_argument('--worker', nargs=3, metavar=('SPOT', 'COUNT', 'SEED'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_case(args.worker[0], args.worker[1], int(args.worker[2]), args.repeats, args.lp_seconds), allow_nan=False))
        return
    import numpy as np
    import scipy
    metadata = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                    numpy=np.__version__, scipy=scipy.__version__)
    for label, key in [('hardware', 'hw.model'), ('memory_bytes', 'hw.memsize'), ('cpu', 'machdep.cpu.brand_string')]:
        try:
            metadata[label] = subprocess.check_output(['/usr/sbin/sysctl', '-n', key], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    results = dict(metadata=metadata, kuhn=kuhn(), cases=[],
                   notice='Exact numerical equilibria in the existing restricted river action tree. Random positive weights on sampled exact combos; full means all 1081 board-compatible combos per player. No neural training. Each LP gets half the stated combined budget. Imports, input generation and process startup are excluded from solver timings; peak RSS includes imports and solver work.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(spot, str(n), seed) for spot in SPOTS for n in args.sizes for seed in args.seeds]
    if args.include_full:
        cases += [(spot, 'full', args.seeds[0]) for spot in SPOTS]
    for spot, count, seed in cases:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', spot, count, str(seed),
                   '--repeats', str(args.repeats), '--lp-seconds', str(args.lp_seconds)]
        wall = time.perf_counter()
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=args.lp_seconds + 60)
            if child.returncode:
                result = dict(spot=spot, count=count, seed=seed, process_error=child.stderr)
            else:
                result = json.loads(child.stdout)
        except subprocess.TimeoutExpired:
            result = dict(spot=spot, count=count, seed=seed, process_error='Whole-process timeout')
        result['subprocess_wall_seconds'] = time.perf_counter() - wall
        results['cases'].append(result)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: result[k] for k in ('spot', 'hands_per_player', 'seed', 'runs', 'process_error') if k in result}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
