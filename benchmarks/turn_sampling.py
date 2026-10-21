"""Equal-time sampled versus enumerated-turn experiment on As Td 6h."""
import argparse
import gc
import json
from pathlib import Path
from heads_up_gto.server import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23])
    parser.add_argument('--max-nodes',type=int,default=2000000)
    parser.add_argument('--output',type=Path,default=Path('turn-sampling-results.json'))
    args=parser.parse_args()
    results={'flop':'As Td 6h','preflop':'raise:2.5/raise:11.5/call','seconds_per_run':args.seconds,
             'note':'Equal time, not equal iterations. Node limits or interruption invalidate the equal-time comparison. Ranges and sizing differ from Wizard.',
             'wizard_observation':{'pot':23,'stacks':[88.5,88.5],'root_actions':{'check':.378,'bet:12.1':.622},
                                   'source':'User-authorized Chrome UI observation, 2026-10-19'},'runs':[]}
    for seed in args.seeds:
        for mode in ('sampled','enumerate_turns'):
            solver,_,metadata=prepare({'flop':results['flop'],'preflop':results['preflop'],
                'time_limit_seconds':args.seconds,'settings':{'mode':'exact','chance_mode':mode,
                'seed':seed,'max_nodes':args.max_nodes or None}})
            stats=solver.train(time_limit_seconds=args.seconds)
            checked=solver.root.apply(solver.root.actions(solver.settings)[0])
            bet=checked.actions(solver.settings)[1].key
            paths={'flop':[],'flop_after_check':['check'], 'turn_checks':['check','check','deal:9h'],
                   'turn_bet_call':['check',bet,'call','deal:9h']}
            views={name:solver.inspect(path) for name,path in paths.items()}
            run={'seed':seed,'chance_mode':mode,'stats':stats,'metadata':metadata,'views':views,
                 'stored':solver.coverage_diagnostics()}
            results['runs'].append(run)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(results,indent=2))
            print(json.dumps({'seed':seed,'mode':mode,'stats':stats,'views':{
                name:{'coverage':v['coverage'],'regret_median':v['training']['regret_updates']['median'],
                      'average_median':v['training']['average_samples']['median'],'aggregate':v['aggregate']}
                for name,v in views.items()}}),flush=True)
            del solver
            gc.collect()

if __name__=='__main__':main()
