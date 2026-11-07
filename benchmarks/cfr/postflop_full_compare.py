"""Whole-postflop DCFR (PostflopCFR) vs b-inary/postflop-solver (the wasm-postflop engine).

Turn-rooted and flop-rooted spots with random weighted ranges. Both solve the same game:
one bet size per street, raises as all-in only ("allin") or no raises ("none"), all-in
always available in "allin" mode. Amounts are integer chips of 0.01 BB; clock excludes
exploitability checks. postflop-solver is native, multithreaded f32; ours uses the compiled numba
engine when installed (--engine numpy for the reference implementation).

Build the Rust CLI once:  cd benchmarks/cfr/postflop_cli && cargo build --release
Run:  PYTHONPATH=. .venv/bin/python benchmarks/cfr/postflop_full_compare.py
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time

from benchmarks.cfr.postflop_compare import CLI, hand_str
from heads_up_gto.cards import COMBOS, card_text, parse_cards, remove_blocked
from heads_up_gto.game import Settings, State
from heads_up_gto.cfr.solver import PostflopCFR

THRESHOLDS = (.005, .002, .001)  # fractions of the pot


def tree(mode, bets):
    """Matching trees: our Settings and postflop-solver's raise string / all-in threshold."""
    if mode == 'allin':
        return (Settings(sizes=tuple((b,) for b in bets), raise_multiplier=1e6, max_raises=2, include_all_in=True),
                'a', 1e6)
    return Settings(sizes=tuple((b,) for b in bets), raise_multiplier=1e6, max_raises=0, include_all_in=False), '', 0.


def run_ours(root, ranges, settings, target, max_seconds, engine='auto'):
    t = time.perf_counter()
    solver = PostflopCFR(root, ranges, settings, max_bytes=12 * 2 ** 30, engine=engine)
    build = time.perf_counter() - t
    solver.solve(iterations=100000, target_exploitability=target * root.pot, check_every=10,
                 time_limit_seconds=max_seconds)
    s = solver.stats
    return dict(build_seconds=build, solve_seconds=s['solve_seconds'], exploitability=s['exploitability_bb'],
                value_bb=s['value_bb'], history=s['history'], memory_gib=s['memory_bytes'] / 2 ** 30,
                decisions=s['public_decisions'], contexts=s['max_contexts'])


def run_theirs(root, ranges, bets, raises, allin, target, max_seconds):
    spec = dict(board=' '.join(card_text(c) for c in root.board), pot=round(root.pot * 100),
                stack=round(root.start_stacks[0] * 100),
                oop=[[hand_str(h), w] for h, w in ranges[0].items()],
                ip=[[hand_str(h), w] for h, w in ranges[1].items()],
                bets=[f'{round(b * 100)}%' for b in bets], raises=raises, add_allin_threshold=allin,
                max_iterations=100000, max_seconds=max_seconds, target_exploitability=target, check_every=10)
    out = subprocess.run([str(CLI)], input=json.dumps(spec), capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    return dict(solve_seconds=r['solve_seconds'], exploitability=r['exploitability'] / 100,
                value_bb=r['avg_ev'][0] / 100 - root.pot / 2, memory_gib=r['memory_bytes'] / 2 ** 30,
                history=[(i, s, e / 100) for i, s, e in r['history']])


def time_to(history, pot):
    return [next((s for _, s, e in history if e <= t * pot), None) for t in THRESHOLDS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--turn-hands', type=int, nargs='*', default=[40, 120, 300])
    p.add_argument('--flop-hands', type=int, nargs='*', default=[20])
    p.add_argument('--mode', choices=('allin', 'none'), default='allin', help='raise handling for turn spots')
    p.add_argument('--target', type=float, default=.001, help='fraction of pot')
    p.add_argument('--max-seconds', type=float, default=300)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--engine', choices=('auto', 'numba', 'numpy'), default='auto', help='engine for our solver')
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/postflop-full-compare.json'))
    args = p.parse_args()
    cases = [('turn', 'As Kd 7c 2h', 1, n, args.mode, (.5, .5, 1.)) for n in args.turn_hands]
    cases += [('flop', 'As Kd 7c', 0, n, 'none', (.5, .5, .5)) for n in args.flop_hands]
    print(f"{'spot':14s} {'hands':>7s} | {'solver':9s} " + ' '.join(f'{t:>8.2%}' for t in THRESHOLDS)
          + '  final%pot  value(BB)  mem GiB')
    results = []
    for street, board, street_index, n, mode, bets in cases:
        settings, raises, allin = (tree(mode, bets) if mode == 'allin' else (*tree(mode, bets),))
        root = State(21, (89.5, 89.5), parse_cards(board), street=street_index)
        pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
        rng = random.Random(args.seed)
        ranges = [{h: round(rng.uniform(.05, 1.), 3) for h in rng.sample(pool, n)} for _ in (0, 1)]
        row = dict(spot=f'{street}/{mode}', hands=n)
        for label, fn in (('ours', lambda: run_ours(root, ranges, settings, args.target, args.max_seconds, args.engine)),
                          ('postflop', lambda: run_theirs(root, ranges, bets, raises, allin, args.target, args.max_seconds))):
            r = fn()
            cells = ' '.join(f"{'   -    ' if t is None else f'{t:7.2f}s'}" for t in time_to(r['history'], root.pot))
            print(f"{row['spot']:14s} {n:3d}x{n:<3d} | {label:9s} {cells}  {r['exploitability'] / root.pot:8.4%} "
                  f"{r['value_bb']:9.4f} {r['memory_gib']:8.2f}", flush=True)
            row[label] = dict(r, time_to=time_to(r['history'], root.pot))
        results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(thresholds=THRESHOLDS, results=results), indent=1) + '\n')


if __name__ == '__main__':
    main()
