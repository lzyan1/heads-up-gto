import unittest
from dataclasses import replace
from heads_up_gto.cards import DECK, parse_cards, parse_range, score
from heads_up_gto.game import Settings, State
from heads_up_gto.solver import Solver


class TurnEnumerationTests(unittest.TestCase):
    def solver(self, **settings):
        return Solver(State(23,(88.5,88.5),parse_cards('As Td 6h')),
                      (parse_range('KsKh'),parse_range('QsQh')),
                      Settings(mode='exact',chance_mode='enumerate_turns',**settings))

    def test_validation(self):
        for config in ({'chance_mode':'bad'}, {'chance_mode':'enumerate_turns','mode':'bucketed'}):
            with self.assertRaises(ValueError):Settings.from_dict(config)
        self.assertEqual(Settings.from_dict({'mode':'exact','chance_mode':'enumerate_turns'}).chance_mode,'enumerate_turns')

    def test_all_45_turns_have_legal_conditional_rivers(self):
        s=self.solver();hands,_=s.sample_deal();bank=s.turn_runouts(hands)
        blocked=set((*hands[0],*hands[1],*s.root.board))
        self.assertEqual({b[3] for b,_ in bank},set(DECK)-blocked)
        self.assertEqual(len(bank),45)
        for board,w in bank:
            self.assertEqual(len(set((*board,*hands[0],*hands[1]))),9)
            a,b=score(hands[0],board),score(hands[1],board)
            self.assertEqual(w,0 if a>b else 1 if b>a else -1)

    def test_single_iteration_visits_each_turn_and_weights_averages(self):
        s=self.solver(sizes=((),(),()),include_all_in=False)
        s.train(1)
        for player in (0,1):
            nodes=[node for key,node in s.nodes.items() if key[0]==player and len(key[3])==4]
            self.assertEqual(len(nodes),45)
            self.assertTrue(all(n.updates==n.averages==1 for n in nodes))
            self.assertTrue(all(abs(sum(n.total)-1/45)<1e-12 for n in nodes))
        self.assertEqual(s.stats()['completed_traversals'],[1,1])

    def test_pending_chance_returns_mean_without_double_weighting(self):
        s=self.solver(sizes=((),(),()),include_all_in=False)
        hands,runout=s.sample_deal();bank=s.turn_runouts(hands)
        pending=s.root.apply(s.root.actions(s.settings)[0])
        pending=pending.apply(pending.actions(s.settings)[0])
        expected=sum(pending.payoff(w,0) for _,w in bank)/len(bank)
        self.assertNotEqual(expected,0)
        self.assertAlmostEqual(s._walk(pending,hands,runout,0,0,turn_bank=bank),expected)

    def test_chance_weight_scales_accumulators_but_not_returned_utility(self):
        a,b=self.solver(),self.solver()
        hands,runout=a.sample_deal()
        state=State(2,(3,3),runout,street=2)
        # Keep the same random stream in both independent traversals.
        b.rng.setstate(a.rng.getstate())
        va=a._walk(state,hands,runout,0,0)
        vb=b._walk(state,hands,runout,0,0,chance_weight=.25)
        self.assertAlmostEqual(va,vb)
        self.assertEqual(a.nodes.keys(),b.nodes.keys())
        for key,na in a.nodes.items():
            nb=b.nodes[key]
            for x,y in zip(na.regrets,nb.regrets):self.assertAlmostEqual(x*.25,y)
            for x,y in zip(na.total,nb.total):self.assertAlmostEqual(x*.25,y)

    def test_flop_allin_uses_mean_and_fold_does_not_use_showdown(self):
        s=self.solver();hands,runout=s.sample_deal();bank=s.turn_runouts(hands)
        terminal=replace(s.root,terminal='showdown',invested=(88.5,88.5))
        expected=sum(terminal.payoff(w,0) for _,w in bank)/45
        self.assertAlmostEqual(s._walk(terminal,hands,runout,0,0,turn_bank=bank),expected)
        folded=replace(terminal,terminal='fold',winner=1)
        self.assertEqual(s._walk(folded,hands,runout,0,0,turn_bank=bank),folded.payoff(1,0))

    def test_resume_and_later_street_refinement(self):
        a,b=self.solver(),self.solver()
        a.train(2);a.train(3);b.train(5)
        self.assertEqual(a.inspect([]),b.inspect([]))
        self.assertEqual(a.enumerated_turns,225)
        a.root=replace(a.root,board=parse_cards('As Td 6h 9h'),street=1)
        self.assertIsNone(a.turn_runouts(a.sample_deal()[0]))
