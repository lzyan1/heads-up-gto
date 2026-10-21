"""Compare the legacy in-place update with frozen policies at equal runtime."""
import argparse,gc,json
from pathlib import Path
from heads_up_gto.server import prepare


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--seconds',type=float,default=30)
 parser.add_argument('--output',type=Path,default=Path('policy-freeze-results.json'))
 args=parser.parse_args();runs=[]
 for seed in (7,23):
  for variant in ('legacy_in_place','simultaneous_frozen'):
   s,_,meta=prepare({'flop':'As Td 6h','preflop':'raise:2.5/raise:11.5/call','time_limit_seconds':args.seconds,
     'settings':{'mode':'exact','chance_mode':'enumerate_turns','seed':seed,'max_nodes':2000000}})
   if variant=='legacy_in_place':
    original=s._walk
    def legacy(state,*a,**kw):
     if state is s.root:s._iteration_strategies=None
     return original(state,*a,**kw)
    s._walk=legacy
   stats=s.train(time_limit_seconds=args.seconds);stats['policy_update']=variant
   root=s.inspect([]);jam=s.inspect(['bet:88.5'])
   counts={'displayed':0,'average_near_pure':0,'current_near_pure':0}
   for hand in s.ranges[0]:
    n=s.nodes.get(s.info_key(s.root,hand))
    if n and n.updates and n.average():
     counts['displayed']+=1
     counts['average_near_pure']+=max(n.average())>=.99
     counts['current_near_pure']+=max(n.strategy())>=.99
   run={'seed':seed,'variant':variant,'stats':stats,'metadata':meta,'root':root,'facing_jam':jam,'purity':counts}
   runs.append(run);args.output.parent.mkdir(parents=True,exist_ok=True)
   args.output.write_text(json.dumps({'note':'Short equal-time regression experiment, not a solved strategy or an exploitability test.','runs':runs},indent=2))
   print(json.dumps({'seed':seed,'variant':variant,'iterations':s.iterations,'root_coverage':root['coverage'],'root_mix':root['aggregate'],'facing_jam_mix':jam['aggregate'],'purity':counts}),flush=True)
   del s
   if variant=='legacy_in_place':del original,legacy
   gc.collect()
if __name__=='__main__':main()
