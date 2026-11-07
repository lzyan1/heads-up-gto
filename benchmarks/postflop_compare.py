"""CFR+ river solver vs b-inary/postflop-solver (the engine inside wasm-postflop).

Both solve the identical river game: bets of 33% and 100% pot, all-in always
available, and a raise only as all-in (so the trees match exactly; verify via
the equal game values). Amounts are integer chips of 0.01 BB. Timings exclude
exploitability checks. postflop-solver is native, multithreaded DCFR in f32;
this solver is single-process numpy CFR+ in f64.

Build the Rust CLI once:  cd benchmarks/postflop_cli && cargo build --release
Run:  PYTHONPATH=. .venv/bin/python benchmarks/postflop_compare.py [--dataset DIR]
"""
import argparse
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from benchmarks.exact_river import SPOTS
from heads_up_gto.cards import COMBOS, card_text, combo_text, parse_cards, remove_blocked
from heads_up_gto.cfr_plus_river import CFRPlusRiver
from heads_up_gto.game import Settings, State

CLI = Path(__file__).parent / 'postflop_cli/target/release/postflop_cli'
BETS = (.33, 1.0)
THRESHOLDS = (.005, .002, .001, .0005)  # fractions of the pot


def settings():
    # raise multiplier so large that the only raise is all-in
    return Settings(sizes=((), (), BETS), raise_multiplier=1e6, max_raises=2, include_all_in=True)


def hand_str(h):
    return ''.join(card_text(c) for c in sorted(h, reverse=True))


def time_to(history, pot, thresholds=THRESHOLDS):
    out = {}
    for t in thresholds:
        hit = next(((s, i) for i, s, e in history if e <= t * pot), None)
        out[f'{t:.2%}'] = None if hit is None else dict(seconds=hit[0], iterations=hit[1])
    return out


def run_ours(root, ranges, target, max_seconds):
    t = time.perf_counter()
    solver = CFRPlusRiver(root, ranges, settings())
    build = time.perf_counter() - t
    solver.solve(iterations=20000, target_exploitability=target * root.pot, check_every=10,
                 time_limit_seconds=max_seconds)
    s = solver.stats
    return dict(build_seconds=build, solve_seconds=s['solve_seconds'], exploitability=s['exploitability_bb'],
                value_bb=s['value_bb'], history=s['history'], nodes=s['public_decisions'])


def run_theirs(root, ranges, target, max_seconds, threads=None):
    pot_c, stack_c = round(root.pot * 100), round(root.start_stacks[0] * 100)
    spec = dict(board=' '.join(card_text(c) for c in root.board), pot=pot_c, stack=stack_c,
                oop=[[hand_str(h), w] for h, w in ranges[0].items()],
                ip=[[hand_str(h), w] for h, w in ranges[1].items()],
                bets=', '.join(f'{round(b * 100)}%' for b in BETS), raises='a',
                add_allin_threshold=1e6, max_iterations=20000, max_seconds=max_seconds,
                target_exploitability=target, check_every=10)
    if threads:
        spec['threads'] = threads
    out = subprocess.run([str(CLI)], input=json.dumps(spec), capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    r['history'] = [(i, s, e / 100) for i, s, e in r['history']]  # chips -> BB
    r['exploitability'] /= 100
    r['avg_ev_bb'] = [v / 100 for v in r['avg_ev']]
    r['value_bb'] = r['avg_ev_bb'][0] - root.pot / 2  # their EV includes the dead-money share
    return r


def spots(args):
    for name, spec in SPOTS.items():
        root = State(spec['pot'], (spec['stack'],) * 2, parse_cards(spec['board']), street=2)
        pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
        for n in args.hands:
            rng = random.Random(args.seed)
            ranges = [{h: round(rng.uniform(.05, 1.), 3) for h in rng.sample(pool, min(n, len(pool)))} for _ in (0, 1)]
            yield f'{name}/{n}', root, ranges
    if args.dataset:
        taken = 0
        for line in (Path(args.dataset) / 'events.jsonl').read_text().splitlines():
            sample = json.loads(line).get('sample')
            if not sample or sum(map(len, sample['ranges'])) < args.min_dataset_hands:
                continue
            root = State(sample['pot_bb'], tuple(sample['stacks_bb']), parse_cards(' '.join(sample['board'])), street=2)
            ranges = [{tuple(parse_cards(' '.join((h[:2], h[2:])))): w for h, w in r.items()} for r in sample['ranges']]
            yield f"dataset/{sample['sample_id']}({sample['formation']})", root, ranges
            taken += 1
            if taken >= args.dataset_spots:
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands', type=int, nargs='+', default=[100, 300, 600])
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', help='Directory with events.jsonl; also benchmark its recorded ranges.')
    p.add_argument('--dataset-spots', type=int, default=4)
    p.add_argument('--min-dataset-hands', type=int, default=600)
    p.add_argument('--target', type=float, default=.0005, help='Stop at this exploitability, fraction of pot.')
    p.add_argument('--max-seconds', type=float, default=60)
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/postflop-compare.json'))
    args = p.parse_args()
    results = []
    header = f"{'spot':32s} {'hands':>9s} | {'solver':14s} " + ' '.join(f'{t:>9.2%}' for t in THRESHOLDS) + '   final%pot   value(BB)'
    print(header)
    for name, root, ranges in spots(args):
        row = dict(spot=name, pot_bb=root.pot, hands=[len(r) for r in ranges], runs={})
        for label, fn in (('ours', lambda: run_ours(root, ranges, args.target, args.max_seconds)),
                          ('postflop', lambda: run_theirs(root, ranges, args.target, args.max_seconds)),
                          ('postflop-1thr', lambda: run_theirs(root, ranges, args.target, args.max_seconds, 1))):
            r = fn()
            row['runs'][label] = dict(r, time_to=time_to(r['history'], root.pot))
            cells = ' '.join(f"{(row['runs'][label]['time_to'][f'{t:.2%}'] or {}).get('seconds', float('nan')):8.2f}s" for t in THRESHOLDS)
            value = r['value_bb']
            print(f"{name:32s} {row['hands'][0]:4d}x{row['hands'][1]:<4d} | {label:14s} {cells}   {r['exploitability'] / root.pot:8.4%}  {value:10.4f}", flush=True)
        results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(machine=platform.platform(), processor=platform.processor(),
                                           bets=BETS, results=results), indent=1) + '\n')


if __name__ == '__main__':
    main()
