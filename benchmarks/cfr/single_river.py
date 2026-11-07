"""CFR+ river solver latency/exploitability on random ranges (same spots as exact_river.py)."""
import argparse
import random
import time

from heads_up_gto.cards import COMBOS, parse_cards, remove_blocked
from heads_up_gto.cfr.single_river import CFRPlusRiver
from heads_up_gto.game import Settings, State
from benchmarks.river.exact_lp import SPOTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands', type=int, nargs='+', default=[100, 300, 600])
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    for name, spec in SPOTS.items():
        root = State(spec['pot'], (spec['stack'],) * 2, parse_cards(spec['board']), street=2,
                     sizing_profile=spec['profile'], aggressor_history=spec['history'])
        pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
        for n in args.hands:
            rng = random.Random(args.seed)
            ranges = [{h: rng.uniform(.05, 1.) for h in rng.sample(pool, min(n, len(pool)))} for _ in (0, 1)]
            t = time.perf_counter()
            s = CFRPlusRiver(root, ranges, Settings()).solve(iterations=args.iterations, check_every=100)
            curve = ' '.join(f'{i}:{e / root.pot:.4%}' for i, _, e in s.stats['history'][::2])
            print(f"{name:20s} n={n:4d} nodes={s.stats['public_decisions']:3d} total={time.perf_counter() - t:6.2f}s "
                  f"solve={s.stats['solve_seconds']:6.2f}s expl(%pot) {curve}", flush=True)


if __name__ == '__main__':
    main()
