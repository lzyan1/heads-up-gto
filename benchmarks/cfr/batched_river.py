"""Throughput of BatchedRiver vs batch size, and per-spot cost vs postflop-solver.

Same river game as postflop_compare.py (bets 33%/100%, raise = all-in). Every
spot in a batch has its own random weighted ranges on one shared board.
Run: PYTHONPATH=. .venv/bin/python benchmarks/cfr/batched_river.py
"""
import argparse
import random
import time

import numpy as np

from benchmarks.river.exact_lp import SPOTS
from benchmarks.cfr.postflop_compare import CLI, run_theirs, settings
from heads_up_gto.cfr.batched_river import BatchedRiver
from heads_up_gto.cards import COMBOS, parse_cards, remove_blocked
from heads_up_gto.game import State


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spot', choices=sorted(SPOTS), default='single_raised_dry')
    p.add_argument('--hands', type=int, default=600)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--target', type=float, default=.0005, help='fraction of pot, applied to the worst spot')
    p.add_argument('--max-seconds', type=float, default=120)
    p.add_argument('--theirs', type=int, default=8, help='postflop-solver spots to time (0 to skip)')
    args = p.parse_args()
    spec = SPOTS[args.spot]
    root = State(spec['pot'], (spec['stack'],) * 2, parse_cards(spec['board']), street=2)
    pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
    rng = random.Random(7)
    spots = [[{h: rng.uniform(.05, 1.) for h in rng.sample(pool, args.hands)} for _ in (0, 1)]
             for _ in range(max(args.batches))]
    print(f'{args.spot}, {args.hands} hands per range, target {args.target:.2%} pot on the worst spot')
    print(f"{'batch':>6s} {'iters':>6s} {'solve s':>9s} {'s/spot':>9s} {'spots/s':>9s} {'worst %pot':>11s} {'mean %pot':>10s}")
    for b in args.batches:
        solver = BatchedRiver(root, spots[:b], settings())
        solver.solve(iterations=5000, target_exploitability=args.target * root.pot, check_every=10,
                     time_limit_seconds=args.max_seconds)
        s = solver.stats
        e = s['exploitability_bb'] / root.pot
        print(f"{b:6d} {s['iterations']:6d} {s['solve_seconds']:9.2f} {s['solve_seconds'] / b:9.3f} "
              f"{b / s['solve_seconds']:9.1f} {e.max():11.4%} {e.mean():10.4%}", flush=True)
    if args.theirs and CLI.exists():
        times = [run_theirs(root, spot, args.target, args.max_seconds)['solve_seconds'] for spot in spots[:args.theirs]]
        print(f'postflop-solver: {np.mean(times):.3f} s/spot over {len(times)} spots (native, multithreaded, per-spot early stop)')


if __name__ == '__main__':
    main()
