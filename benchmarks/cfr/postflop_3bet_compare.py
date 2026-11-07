"""Real 3-bet-pot flop (chart ranges) solved by PostflopCFR and by postflop-solver on the same tree.

Pot 23 BB, 88.5 BB stacks, GTO Wizard chart ranges after SB raises to 2.5, BB 3-bets to 11.5, SB calls.
Both solvers use the same generic tree: flop/turn/river bet sizes below for either player, raises
are all-in only, all-in always available. (postflop-solver cannot express the app's scenario menu.)
Run: PYTHONPATH=. .venv/bin/python benchmarks/cfr/postflop_3bet_compare.py [--board 'As Kd 7c']
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from benchmarks.cfr.postflop_compare import CLI, hand_str
from heads_up_gto.cards import card_text, parse_cards
from heads_up_gto.game import Settings, State
from heads_up_gto.cfr.solver import PostflopCFR
from heads_up_gto.preflop import chart_ranges, reconstruct
from heads_up_gto.screenshot_charts import screenshot_chart

THRESHOLDS = (.01, .005, .002, .001)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board', default='As Kd 7c')
    p.add_argument('--bets', nargs=3, default=['33', '66', '33,100'], help='percent of pot per street (comma separated)')
    p.add_argument('--target', type=float, default=.001)
    p.add_argument('--max-seconds', type=float, default=900)
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/postflop-3bet-compare.json'))
    args = p.parse_args()
    board = parse_cards(args.board, 3)
    pre = reconstruct('raise:2.5/raise:11.5/call')
    ranges = chart_ranges(screenshot_chart(), pre, board)
    sizes = [tuple(float(x) / 100 for x in street.split(',')) for street in args.bets]
    settings = Settings(sizes=tuple(sizes), raise_multiplier=1e6, max_raises=2, include_all_in=True)
    root = State(pre.pot, pre.stacks, board)
    print(f'3-bet pot {root.pot} BB, stacks {root.stacks}, board {args.board}, combos {[len(r) for r in ranges]}, '
          f'bets {args.bets}', flush=True)
    results = {}

    t = time.perf_counter()
    solver = PostflopCFR(root, ranges, settings, max_bytes=12 * 2 ** 30)
    solver.allocate()
    build = time.perf_counter() - t
    solver.solve(iterations=10 ** 6, target_exploitability=args.target * root.pot, check_overhead=.15,
                 check_every=10 ** 9, time_limit_seconds=args.max_seconds)
    s = solver.stats
    results['ours'] = dict(build_seconds=build, solve_seconds=s['solve_seconds'], iterations=s['iterations'],
                           exploitability=s['exploitability_bb'], value_bb=s['value_bb'],
                           history=s['history'], memory_gib=s['memory_bytes'] / 2 ** 30, hands=solver.size)

    spec = dict(board=args.board, pot=round(root.pot * 100), stack=round(root.start_stacks[0] * 100),
                oop=[[hand_str(h), w] for h, w in ranges[0].items()],
                ip=[[hand_str(h), w] for h, w in ranges[1].items()],
                bets=[','.join(f'{x}%' for x in street.split(',')) for street in args.bets], raises='a',
                add_allin_threshold=1e6, max_iterations=10 ** 6, max_seconds=args.max_seconds,
                target_exploitability=args.target, check_every=10)
    out = subprocess.run([str(CLI)], input=json.dumps(spec), capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    results['postflop'] = dict(solve_seconds=r['solve_seconds'], exploitability=r['exploitability'] / 100,
                               value_bb=r['avg_ev'][0] / 100 - root.pot / 2, memory_gib=r['memory_bytes'] / 2 ** 30,
                               history=[(i, sec, e / 100) for i, sec, e in r['history']],
                               iterations=r['history'][-1][0] if r['history'] else 0)

    print(f"{'solver':10s} " + ' '.join(f'{t:>9.2%}' for t in THRESHOLDS) + '  iters  final%pot  value(BB)  memGiB')
    for name, res in results.items():
        times = [next((sec for _, sec, e in res['history'] if e <= t * root.pot), None) for t in THRESHOLDS]
        cells = ' '.join('        -' if v is None else f'{v:8.1f}s' for v in times)
        print(f"{name:10s} {cells}  {res['iterations']:5d}  {res['exploitability'] / root.pot:8.4%}  "
              f"{res['value_bb']:9.4f}  {res['memory_gib']:6.2f}")
        res['time_to'] = times
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(board=args.board, pot=root.pot, results=results), indent=1) + '\n')


if __name__ == '__main__':
    main()
