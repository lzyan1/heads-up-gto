import itertools
import random
import unittest
from heads_up_gto.cards import parse_cards, parse_range, score
from heads_up_gto.preflop import reconstruct, chart_ranges, demo_chart, validate_chart
from heads_up_gto.game import State, Settings
from heads_up_gto.range_game import RangeGame


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


class RangeSamplingTests(unittest.TestCase):
    def test_sampling_correct_blocker_weighted_joint(self):
        root=State(2,(99,99),cards('2c 7d 9h'));r0=parse_range('AsAh, KsKh');r1=parse_range('AsAd, QsQh')
        solver=RangeGame(root,(r0,r1),Settings());count={h:0 for h in r0}
        for _ in range(9000):
            hands,runout=solver.sample_deal();count[hands[0]]+=1
            self.assertEqual(len(set((*hands[0],*hands[1],*runout))),9)
        self.assertAlmostEqual(count[tuple(sorted(cards('AsAh')))]/9000,1/3,delta=.02)

    def test_ranges_need_a_compatible_private_deal(self):
        root = State(2, (3, 3), cards('2c 7d 9h'))
        for ranges in (({}, parse_range('AA')), (parse_range('AsAh'), parse_range('AsAd'))):
            with self.assertRaises(ValueError):
                RangeGame(root, ranges, Settings())


if __name__=='__main__':unittest.main()
