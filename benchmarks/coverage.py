"""Reproduce exact-hand coverage on a fixed flop with all-in actions enabled."""
import argparse
import json
from pathlib import Path
from heads_up_gto.server import prepare

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('coverage-results.json'))
args=parser.parse_args()
s,_,meta=prepare({'flop':'8s 7h 2d','preflop':'raise:2.5/call','settings':{'mode':'exact'},'iterations':130000})
results={'configuration':meta,'checkpoints':[]}
checked=s.root.apply(s.root.actions(s.settings)[0])
cbet=checked.actions(s.settings)[1].key
for target in (1000,10000,50000,130000):
    stats=s.train(target-s.iterations)
    views={}
    for name,path in {'flop':[], 'turn_checks':['check','check','deal:Ac'], 'turn_bet_call':['check',cbet,'call','deal:Ac']}.items():
        v=s.inspect(path)
        views[name]={'coverage':v['coverage'],'training':v['training'],'fallback_used':v['fallback_used'], 'bottom_sets':[r for r in v['rows'] if r['class']=='22']}
    checkpoint={'stats':stats,'views':views,'stored':s.coverage_diagnostics()}
    results['checkpoints'].append(checkpoint)
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps({'stats':stats,'coverage':{k:v['coverage'] for k,v in views.items()}}),flush=True)
