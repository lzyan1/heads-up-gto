"""Equal-time tabular/Deep CFR comparison; exact exploitability on tiny rivers.

Run from the repository with PYTHONPATH=. and the optional .[deep] dependencies.
Turn results report sampling/predictions only, not an exploitability estimate.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

from heads_up_gto.backends import create_solver
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.diagnostics import river_exploitability
from heads_up_gto.game import Settings, State


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 23])
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    parser.add_argument('--street', choices=['turn', 'river'], default='river')
    parser.add_argument('--output', type=Path, default=Path('deep-cfr-results.json'))
    args = parser.parse_args()
    board = parse_cards('2c 7d 9h Js' + (' Qc' if args.street == 'river' else ''))
    root = State(2, (3, 3), board, street=len(board) - 3)
    ranges = parse_range('AsAh,6s6h'), parse_range('KsKh,5s5h')
    results = {'platform': platform.platform(), 'street': args.street,
               'requested_seconds_per_run': args.seconds, 'runs': [],
               'notice': 'Small diagnostic game only; these results do not establish full-game strength.'}
    try:
        import torch
        results['torch_version'] = torch.__version__
        results['mps_available'] = torch.backends.mps.is_available()
    except ImportError:
        pass
    for seed in args.seeds:
        for backend, device in [('tabular', 'cpu'), *(('deep_cfr', d) for d in args.devices)]:
            settings = Settings(mode='exact', backend=backend, deep_device=device, seed=seed)
            record = {'backend': backend, 'device': device, 'seed': seed, 'settings': asdict(settings)}
            try:
                solver = create_solver(root, ranges, settings)
            except ValueError as exc:
                record['unavailable'] = str(exc)
            else:
                if args.street == 'river':
                    record['initial_exploitability_bb'] = river_exploitability(solver)['exploitability_bb']
                record['stats'] = solver.train(time_limit_seconds=args.seconds)
                if args.street == 'river':
                    record['exploitability_bb'] = river_exploitability(solver)['exploitability_bb']
                paths = {'root': []}
                if args.street == 'turn':
                    paths['river_after_checks'] = ['check', 'check', 'deal:Qc']
                record['views'] = {}
                for name, path in paths.items():
                    view = solver.inspect(path)
                    record['views'][name] = {key: view.get(key) for key in
                                             ('coverage', 'prediction_coverage', 'training', 'aggregate')}
            results['runs'].append(record)
            args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
            print(json.dumps({k: v for k, v in record.items() if k not in ('settings', 'views')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
