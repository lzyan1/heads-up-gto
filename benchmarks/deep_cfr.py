"""Hybrid earlier-street training CPU/GPU comparison with direct river boundaries.

Run from the repository with PYTHONPATH=. and the project dependencies.
Flop/turn results report sampling/predictions only, not an exploitability estimate.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

from heads_up_gto.deep_cfr import DeepCFRSolver
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 23])
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['cpu'])
    parser.add_argument('--street', choices=['flop', 'turn'], default='turn')
    parser.add_argument('--output', type=Path, default=Path('deep-cfr-results.json'))
    args = parser.parse_args()
    board = parse_cards('2c 7d 9h Js Qc')[:{'flop': 3, 'turn': 4, 'river': 5}[args.street]]
    root = State(2, (3, 3), board, street=len(board) - 3)
    ranges = parse_range('AsAh,6s6h'), parse_range('KsKh,5s5h')
    results = {'platform': platform.platform(), 'street': args.street,
               'requested_seconds_per_run': args.seconds, 'runs': [],
               'notice': 'Small diagnostic game only; these results do not establish full-game strength.'}
    import torch
    results['torch_version'] = torch.__version__
    results['mps_available'] = torch.backends.mps.is_available()
    for seed in args.seeds:
        for device in args.devices:
            settings = Settings(deep_device=device, seed=seed)
            record = {'backend': 'deep_cfr', 'device': device, 'seed': seed, 'settings': asdict(settings)}
            try:
                solver = DeepCFRSolver(root, ranges, settings)
            except ValueError as exc:
                record['unavailable'] = str(exc)
            else:
                record['stats'] = solver.train(time_limit_seconds=args.seconds)
                paths = {'root': []}
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
