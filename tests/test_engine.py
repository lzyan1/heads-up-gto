import itertools
import random
import unittest
from unittest.mock import patch
from heads_up_gto.cards import parse_cards, parse_range, score
from heads_up_gto.preflop import reconstruct, chart_ranges, demo_chart, validate_chart
from heads_up_gto.game import State, Settings
from heads_up_gto.solver import Solver, Node
from heads_up_gto.diagnostics import river_exploitability


def cards(text): return parse_cards(text)
def act(state,key,settings=Settings()): return state.apply(next(a for a in state.actions(settings) if a.key==key))


class CardsTests(unittest.TestCase):
    def test_parsing_and_duplicates(self):
        self.assertEqual(cards('A♠ 10♥ 7c'),cards('As Th 7c'))
        for text in ('As As','Asxx','Zs Kd','AsK'):
            with self.assertRaises(ValueError):cards(text)

    def test_range_counts_and_weights(self):
        for text,count in [('AA',6),('AKs',4),('AKo',12),('AK',16),('*',1326)]:self.assertEqual(len(parse_range(text)),count)
        r=parse_range('AA:0.5, AsAh:0.25')
        self.assertEqual(r[tuple(sorted(cards('AsAh')))],.25)
        for value in ('AA:2','AKs:nan','TT+','AA:-1'):
            with self.assertRaises(ValueError):parse_range(value)

    def test_hand_ranking_categories(self):
        examples=['As Kd 9h 7c 4s','As Ad 9h 7c 4s','As Ad 9h 9c 4s','As Ad Ah 7c 4s','As 2d 3h 4c 5s','As Ts 9s 7s 4s','As Ad Ah 7c 7s','As Ad Ah Ac 4s','As Ks Qs Js Ts']
        scores=[score((),cards(x)) for x in examples]
        self.assertEqual(scores,sorted(scores))
        self.assertEqual([s[0] for s in scores],list(range(9)))

    def test_double_trips_and_flush_kickers(self):
        self.assertEqual(score((),cards('As Ad Ah Ks Kd Kh 2c')),(6,14,13))
        self.assertGreater(score((),cards('As Qs 9s 7s 3s')),score((),cards('As Js 9s 7s 3s')))
        self.assertGreater(score((),cards('2s 3d 4h 5c 6s')),score((),cards('As 2d 3h 4c 5s')))

    def test_seven_equals_best_five(self):
        rng=random.Random(42)
        for _ in range(100):
            hand=tuple(rng.sample(range(52),7))
            self.assertEqual(score((),hand),max(score((),five) for five in itertools.combinations(hand,5)))


class PreflopTests(unittest.TestCase):
    def test_pots_positions(self):
        self.assertEqual(reconstruct('raise:2.5/call').pot,5)
        self.assertEqual(reconstruct('call/check').stacks,(99,99))
        self.assertEqual(reconstruct('raise:2.5/raise:9/call').pot,18)
        self.assertEqual(reconstruct('raise:2.5/raise:9/raise:22/call').stacks,(78,78))

    def test_illegal_and_incomplete_histories(self):
        for line in ('check','raise:1.5/call','raise:2.5/check','raise:2.5','raise:2.5/raise:3/call','call/check/check','fold','raise:100/call'):
            with self.subTest(line=line),self.assertRaises(ValueError):reconstruct(line)

    def test_chart_multiplication_and_blockers(self):
        chart={'version':1,'game':demo_chart()['game'],'nodes':{'':{'raise:2.5':{'AA':.5}},'raise:2.5':{'raise:9':{'KK':.25}},'raise:2.5/raise:9':{'call':{'AA':.4}}}}
        bb,sb=chart_ranges(chart,reconstruct('raise:2.5/raise:9/call'),cards('As 7d 2c'))
        self.assertEqual(len(sb),3)
        self.assertTrue(all(abs(w-.2)<1e-8 for w in sb.values()))
        self.assertEqual(len(bb),6)
        self.assertTrue(all(w==.25 for w in bb.values()))

    def test_chart_checks(self):
        validate_chart(demo_chart())
        chart=demo_chart();chart['game']['rake']=.05
        with self.assertRaises(ValueError):validate_chart(chart)
        chart=demo_chart();chart['nodes']['']={'call':{'AA':.6},'raise:2.5':{'AA':.6}}
        with self.assertRaises(ValueError):validate_chart(chart)
        with self.assertRaises(ValueError):chart_ranges(demo_chart(),reconstruct('raise:3/call'),cards('As Kd 7c'))


class BettingTests(unittest.TestCase):
    def setUp(self):self.root=State(5,(97.5,97.5),cards('As Kd 7c'))

    def test_check_check_and_positions(self):
        s=act(act(self.root,'check'),'check');self.assertTrue(s.pending)
        s=s.deal(cards('2h')[0])
        self.assertEqual((s.actor,s.street,s.pot),(0,1,5))
        self.assertEqual(s.paid,(0,0));self.assertFalse(s.checked)

    def test_bet_raise_call(self):
        s=act(act(self.root,'bet:3.75'),'raise:11.25')
        self.assertEqual([a.key for a in s.actions(Settings())],['fold','call'])
        s=act(s,'call')
        self.assertEqual(s.pot,27.5);self.assertEqual(s.stacks,(86.25,86.25));self.assertTrue(s.pending)

    def test_allin_and_payoff(self):
        s=act(self.root,'bet:97.5')
        self.assertEqual([a.key for a in s.actions(Settings())],['fold','call'])
        fold=act(s,'fold')
        self.assertEqual(fold.payoff(fold.winner,0),2.5);self.assertEqual(fold.payoff(fold.winner,1),-2.5)
        call=act(s,'call')
        self.assertEqual(call.pot,200);self.assertEqual(call.terminal,'showdown')
        self.assertEqual(call.payoff(0,0),100);self.assertEqual(call.payoff(-1,0),0)

    def test_short_stack_and_min_raise(self):
        s=State(199,(.5,.5),cards('As Kd 7c'))
        self.assertEqual([a.key for a in s.actions(Settings())],['check','bet:0.5'])
        s=act(s,'bet:0.5');self.assertEqual([a.key for a in s.actions(Settings())],['fold','call'])

    def test_invariants_random_lines(self):
        rng=random.Random(3)
        for _ in range(80):
            s=self.root
            while not s.terminal:
                if s.pending:
                    s=s.deal(next(c for c in range(52) if c not in s.board));continue
                s=s.apply(rng.choice(s.actions(Settings())))
                self.assertAlmostEqual(s.pot+sum(s.stacks),200);self.assertGreaterEqual(min(s.stacks),0)
            for winner in (-1,0,1):self.assertAlmostEqual(s.payoff(winner,0)+s.payoff(winner,1),0)


class SolverTests(unittest.TestCase):
    def small(self,mode='exact'):
        root=State(2,(3,3),cards('2c 7d 9h Js Qc'),street=2)
        ranges=(parse_range('AsAh, 6s6h'),parse_range('KsKh, 5s5h'))
        settings=Settings(sizes=((.5,),(.5,),(.5,)),include_all_in=False,max_raises=0,mode=mode,buckets=4,equity_samples=8,seed=42)
        return Solver(root,ranges,settings)

    def test_exact_river_converges_against_best_response(self):
        solver=self.small();initial=river_exploitability(solver)['exploitability_bb'];solver.train(6000)
        final=river_exploitability(solver)['exploitability_bb']
        self.assertLess(final,.04);self.assertLess(final,initial*.2)

    def test_reproducible_and_resume(self):
        a,b=self.small(),self.small();a.train(80);a.train(120);b.train(200)
        self.assertEqual(a.inspect([]),b.inspect([]))

    def test_timed_runs_finish_iterations_and_resume_with_a_fresh_deadline(self):
        solver, baseline = self.small(), self.small()
        solver.train(10)
        previous = solver.elapsed
        clock = [100.0]
        sample = solver.sample_deal
        reports = []
        def slow_deal():
            clock[0] += .4
            return sample()
        with patch('heads_up_gto.solver.time.monotonic', side_effect=lambda: clock[0]), patch.object(solver, 'sample_deal', side_effect=slow_deal):
            first = solver.train(time_limit_seconds=.5, progress=reports.append)
            second = solver.train(time_limit_seconds=.5)
        self.assertEqual((first['iterations'], second['iterations']), (12, 14))
        self.assertEqual(first['status'], 'complete')
        self.assertIsNone(first['target'])
        self.assertEqual(first['time_limit_seconds'], .5)
        self.assertEqual((first['run_seconds'], second['run_seconds']), (.8, .8))
        self.assertAlmostEqual(solver.elapsed, previous + 1.6)
        self.assertEqual(reports[-1]['run_seconds'], .8)
        baseline.train(14)
        self.assertEqual(solver.inspect([]), baseline.inspect([]))
        resumed = solver.train(10)
        self.assertEqual(resumed['target'], 24)
        self.assertIsNone(resumed['time_limit_seconds'])

    def test_time_limit_validation_and_manual_stop(self):
        solver = self.small()
        for kwargs in ({}, {'iterations': 10, 'time_limit_seconds': 1}, {'time_limit_seconds': 0}, {'time_limit_seconds': -1}, {'time_limit_seconds': float('nan')}, {'time_limit_seconds': float('inf')}, {'time_limit_seconds': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.train(**kwargs)
        solver.cancel.set()
        result = solver.train(time_limit_seconds=10)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['iterations'], 0)

    def test_average_probabilities(self):
        solver=self.small();solver.train(200);result=solver.inspect([])
        self.assertAlmostEqual(sum(result['aggregate']),1)
        self.assertAlmostEqual(sum(r['mass'] for r in result['rows']),1)
        for r in result['rows']:self.assertAlmostEqual(sum(r['probabilities']),1)

    def test_never_fabricate_untrained_recommendations(self):
        solver=self.small();view=solver.inspect([])
        self.assertEqual(view['coverage'],0);self.assertTrue(all(r['probabilities'] is None for r in view['rows']))
        self.assertTrue(solver.inspect(['check'])['fallback_used'])
        hand=next(iter(solver.ranges[0]));key=solver.info_key(solver.root,hand)
        fresh=Node(len(solver.root.actions(solver.settings)));fresh.total=[1.0]*len(fresh.total);fresh.averages=1;solver.nodes[key]=fresh
        self.assertIsNone(solver.policy(solver.root,hand)[0])

    def test_sampling_correct_blocker_weighted_joint(self):
        root=State(2,(99,99),cards('2c 7d 9h'));r0=parse_range('AsAh, KsKh');r1=parse_range('AsAd, QsQh')
        solver=Solver(root,(r0,r1),Settings());count={h:0 for h in r0}
        for _ in range(9000):
            hands,runout=solver.sample_deal();count[hands[0]]+=1
            self.assertEqual(len(set((*hands[0],*hands[1],*runout))),9)
        self.assertAlmostEqual(count[tuple(sorted(cards('AsAh')))]/9000,1/3,delta=.02)

    def test_no_future_information_leak_and_recall(self):
        solver=self.small('bucketed');root=State(2,(99,99),cards('2c 7d 9h'));solver.root=root
        hand=next(iter(solver.ranges[0]));k0=solver.info_key(root,hand)
        turn=act(act(root,'check'),'check').deal(cards('Ts')[0]);k1=solver.info_key(turn,hand)
        self.assertEqual(k0[2][0],k1[2][0]);self.assertEqual(len(k1[2]),2)

    def test_tree_budget_returns_partial_state(self):
        solver=self.small();solver.settings=Settings(mode='exact',max_nodes=1)
        self.assertEqual(solver.train(50)['status'],'node_limit')

    def test_real_memory_exhaustion_retains_results_and_can_resume(self):
        solver = self.small()
        solver.train(25)
        before = solver.inspect([])
        with patch.object(solver, 'sample_deal', side_effect=MemoryError):
            result = solver.train(25)
        self.assertEqual(result['status'], 'memory_exhausted')
        self.assertEqual(result['iterations'], 25)
        self.assertEqual(before, solver.inspect([]))
        self.assertEqual(solver.train(25)['status'], 'complete')
        self.assertEqual(solver.iterations, 50)

    def test_blocked_selected_hand(self):
        solver=self.small();solver.train(20)
        self.assertIn('selected_error',solver.inspect([],cards('2c As')))
        with self.assertRaises(ValueError):solver.inspect(['bet:9999'])


if __name__=='__main__':unittest.main()
