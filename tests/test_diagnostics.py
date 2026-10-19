import itertools
import random
import unittest
from dataclasses import dataclass
from heads_up_gto.game import Action, Settings, State
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.solver import Solver, Node
from heads_up_gto.diagnostics import sample_summary, river_exploitability


@dataclass(frozen=True)
class Kuhn:
    path: tuple = ()
    board: tuple = ()
    pending: bool = False

    @property
    def actor(self): return len(self.path) % 2
    @property
    def terminal(self):
        if self.path and self.path[-1] == 'fold': return 'fold'
        return 'showdown' if self.path in (('check','check'),('bet','call'),('check','bet','call')) else ''
    @property
    def winner(self): return self.actor
    def actions(self, settings):
        names = ('fold','call') if self.path and self.path[-1]=='bet' else ('check','bet')
        return tuple(Action(x,x) for x in names)
    def apply(self, action): return Kuhn((*self.path,action.key))
    def payoff(self, winner, player):
        stake = 2 if self.path[-1]=='call' else 1
        return stake if winner==player else -stake


def kuhn_experiment(iterations=20000):
    solver=Solver(State(2,(3,3),parse_cards('2c 7d 9h Js Qc'),street=2),
                  (parse_range('AsAh'),parse_range('KsKh')),Settings(mode='exact',seed=42))
    root=Kuhn()
    rng=random.Random(42)
    deals=list(itertools.permutations(range(3),2))
    for _ in range(iterations):
        hands=rng.choice(deals)
        winner=int(hands[1]>hands[0])
        for player in (0,1):solver._walk(root,hands,(),winner,player)
    def evaluate(hands, pure_player=None, pure=None):
        def walk(state):
            if state.terminal:return state.payoff(state.winner if state.terminal=='fold' else int(hands[1]>hands[0]),0)
            actions=state.actions(None)
            if state.actor==pure_player:
                stage=0 if len(state.path)<2 else 1
                # P1 decisions: after check versus after bet.
                if pure_player==1:stage=int(state.path==('bet',))
                return walk(state.apply(actions[pure[2*hands[state.actor]+stage]]))
            probs=solver.policy(state,hands[state.actor])[0] or [.5,.5]
            return sum(p*walk(state.apply(a)) for p,a in zip(probs,actions))
        return walk(root)
    value=sum(evaluate(h) for h in deals)/6
    br0=max(sum(evaluate(h,0,p) for h in deals)/6 for p in itertools.product(range(2),repeat=6))
    br1=min(sum(evaluate(h,1,p) for h in deals)/6 for p in itertools.product(range(2),repeat=6))
    return {'iterations':iterations,'value':value,'exploitability':(br0-br1)/2}


class DiagnosticTests(unittest.TestCase):
    def solver(self, **kwargs):
        return Solver(State(2,(3,3),parse_cards('2c 7d 9h Js Qc'),street=2),
                      (parse_range('AsAh,6s6h'),parse_range('KsKh,5s5h')),Settings(mode='exact',**kwargs))

    def test_sample_states_and_threshold_are_separate(self):
        rows=[{'updates':u,'averages':a,'mass':.2} for u,a in [(0,0),(0,2),(3,0),(1,900),(200,200)]]
        summary=sample_summary(rows)
        self.assertTrue(all(v==1 for v in summary['counts'].values()))
        self.assertAlmostEqual(sum(summary['mass'].values()),1)
        self.assertEqual(summary['regret_updates']['median'],1)

    def test_interruption_is_visible_and_resume_keeps_counters(self):
        s=self.solver(max_nodes=1)
        stats=s.train(10)
        self.assertEqual(stats['iterations'],0)
        self.assertEqual(stats['attempted_iterations'],1)
        self.assertEqual(stats['interrupted_iterations'],1)
        self.assertTrue(stats['partial_updates_possible'])
        s.settings=Settings(mode='exact')
        stats=s.train(10)
        self.assertEqual(stats['iterations'],10)
        self.assertEqual(stats['attempted_iterations'],11)
        self.assertEqual(stats['completed_traversals'],[10,10])

    def test_inspection_diagnostics_and_selected_mass(self):
        s=self.solver();s.train(100)
        hand=next(iter(s.ranges[0]))
        view=s.inspect([],hand)
        self.assertEqual(view['selected'],next(r for r in view['rows'] if r['hand']==view['selected']['hand']))
        self.assertAlmostEqual(sum(view['training']['mass'].values()),1)
        self.assertEqual(sum(x['hands'] for x in s.coverage_diagnostics()['streets'].values()),len(s.nodes))

    def test_actual_walk_matches_kuhn_equilibrium(self):
        result=kuhn_experiment()
        self.assertAlmostEqual(result['value'],-1/18,delta=.003)
        self.assertLess(result['exploitability'],.02)

    def test_river_with_allins_converges(self):
        for seed in (7,42):
            s=self.solver(seed=seed)
            initial=river_exploitability(s)['exploitability_bb']
            s.train(10000)
            final=river_exploitability(s)['exploitability_bb']
            self.assertLess(final,.06)
            self.assertLess(final,initial*.1)
            self.assertIn('bet:3',[a.key for a in s.root.actions(s.settings)])
