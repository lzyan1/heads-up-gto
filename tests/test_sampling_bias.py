import itertools
import unittest
from dataclasses import dataclass
from heads_up_gto.game import State,Settings,Action
from heads_up_gto.solver import Solver
from heads_up_gto.cards import parse_cards,parse_range
@dataclass(frozen=True)
class CoinBet:
 path:tuple=()
 board:tuple=()
 pending:bool=False
 @property
 def actor(self):return len(self.path)%2
 @property
 def terminal(self):return 'fold' if self.path and self.path[-1]=='fold' else 'showdown' if self.path and self.path[-1] in ('check','call') else ''
 @property
 def winner(self):return 0
 def actions(self,settings):return tuple(Action(x,x) for x in (('check','bet') if not self.path else ('fold','call')))
 def apply(self,a):return CoinBet((*self.path,a.key))
 def payoff(self,w,p):
  v=0 if self.path[-1]=='check' else 1 if self.path[-1]=='fold' else 3 if w==0 else -3
  return v if p==0 else -v

class Controlled:
 def __init__(self,us):self.us=iter(us)
 def choices(self,actions,weights,k):
  threshold=next(self.us)*sum(weights);total=0
  for a,w in zip(actions,weights):
   total+=w
   if threshold<total:return [a]
  return [actions[-1]]
def expected(frozen):
 total=0
 for w,u,v in itertools.product((0,1),(.25,.75),(.25,.75)):
  s=Solver(State(2,(3,3),parse_cards('As Td 6h')),(parse_range('KsKh'),parse_range('QsQh')),Settings(mode='exact'))
  s.rng=Controlled((u,v))
  s._iteration_strategies={} if frozen else None
  s._walk(CoinBet(),(0,1),(),w,0)
  s._walk(CoinBet(),(0,1),(),w,1)
  total+=s.nodes[s.info_key(CoinBet(('bet',)),1)].regrets[1]/8
 return total

class SamplingBiasTests(unittest.TestCase):
 def test_shared_chance_requires_frozen_policy(self):
  # Hidden fair coin: checking returns 0; a bet wins 1 on a fold,
  # or +/-3 on a call. Neither player observes the coin. Under the
  # initial uniform profile: P(bet)=1/2, call regret=1/2 => E=1/4.
  # Enumerate every coin and sampled action, without statistical error.
  self.assertAlmostEqual(expected(False),0.0)
  self.assertAlmostEqual(expected(True),.25)

 def test_train_freezes_both_passes_and_releases_cache(self):
  from unittest.mock import patch
  s=Solver(State(2,(3,3),parse_cards('As Td 6h')),(parse_range('KsKh'),parse_range('QsQh')),Settings(mode='exact'))
  seen=[]
  walk=s._walk
  def inspect(state,*args,**kwargs):
   if state is s.root:seen.append(s._iteration_strategies)
   return walk(state,*args,**kwargs)
  with patch.object(s,'_walk',side_effect=inspect):s.train(2)
  self.assertIs(seen[0],seen[1])
  self.assertIs(seen[2],seen[3])
  self.assertIsNot(seen[0],seen[2])
  self.assertTrue(all(cache for cache in seen))
  self.assertIsNone(s._iteration_strategies)
  s.settings=Settings(mode='exact',max_nodes=1)
  s.train(100)
  self.assertIsNone(s._iteration_strategies)
