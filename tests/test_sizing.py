import unittest
from dataclasses import replace
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State
from heads_up_gto.server import prepare
from heads_up_gto.sizing import board_features, choose_size, opening_scenario, profile_for
from heads_up_gto.solver import Solver


class ScenarioSizingTests(unittest.TestCase):
    settings = Settings()

    def root(self, pot=5, stack=97.5, board="As Kd 7c"):
        return State(pot, (stack, stack), parse_cards(board), sizing_profile=profile_for(pot, (stack, stack)))

    def act(self, state, key):
        return state.apply(next(a for a in state.actions(self.settings) if a.key == key))

    def advance(self, state, line, card):
        for key in line:
            state = self.act(state, key)
        return state.deal(parse_cards(card)[0])

    def keys(self, state):
        return [a.key for a in state.actions(self.settings)]

    def test_flop_examples_keep_contextual_size_alongside_jam(self):
        dry, wet = self.root(), self.root(board="9h 8h 6c")
        self.assertEqual(self.keys(dry), ["check", "bet:3.8", "bet:97.5"])
        self.assertEqual(self.keys(wet), ["check", "bet:5", "bet:97.5"])
        self.assertEqual(self.keys(self.act(dry, "check")), ["check", "bet:1.7", "bet:97.5"])
        self.assertEqual(self.keys(self.act(wet, "check")), ["check", "bet:3.4", "bet:97.5"])
        three = self.root(23, 88.5)
        self.assertEqual(choose_size(three).scenario, "cbet")
        self.assertEqual(self.keys(three), ["check", "bet:7.6", "bet:88.5"])
        four = self.root(52, 74)
        self.assertEqual(self.keys(four), ["check", "bet:26", "bet:74"])
        self.assertEqual(self.keys(self.act(four, "check")), ["check", "bet:13", "bet:74"])

    def test_short_stacks_and_raises_use_one_jam(self):
        state = self.root()
        for key in ("check", "bet:1.7", "raise:8"):
            state = self.act(state, key)
        self.assertEqual(self.keys(state), ["fold", "call", "raise:97.5"])
        state = self.act(state, "raise:97.5")
        self.assertEqual(self.keys(state), ["fold", "call"])
        self.assertEqual(self.act(state, "call").terminal, "showdown")
        short = replace(self.root(52, 74), start_stacks=(.35, 1.))
        self.assertEqual(self.keys(short), ["check", "bet:0.35"])
        self.assertEqual(self.keys(self.act(short, "bet:0.35")), ["fold", "call"])

    def test_history_distinguishes_donk_barrel_probe_and_stab(self):
        turn = self.advance(self.root(), ("check", "bet:1.7", "call"), "2h")
        self.assertEqual(turn.aggressor_history, (1,))
        self.assertEqual(opening_scenario(turn), "donk")
        self.assertEqual(opening_scenario(self.act(turn, "check")), "barrel")
        checked = self.advance(self.root(), ("check", "check"), "2h")
        self.assertEqual(opening_scenario(checked), "probe")
        self.assertEqual(opening_scenario(self.act(checked, "check")), "delayed_cbet")
        oop = self.advance(self.root(23, 88.5), ("bet:7.6", "call"), "2h")
        self.assertEqual(opening_scenario(oop), "barrel")
        self.assertEqual(opening_scenario(self.act(oop, "check")), "stab")

    def test_new_card_changes_menu_and_refinement_keeps_context(self):
        flop = self.root(board="9h 8h 6c")
        pending = self.act(self.act(self.act(flop, "check"), "bet:3.4"), "call")
        blank = self.act(pending.deal(parse_cards("2s")[0]), "check")
        flush = self.act(pending.deal(parse_cards("2h")[0]), "check")
        self.assertNotEqual(self.keys(blank), self.keys(flush))
        for state in (blank, flush):
            refined = replace(state, path=())
            self.assertEqual(refined.actions(self.settings), state.actions(self.settings))
            self.assertEqual(choose_size(refined), choose_size(state))
        river = self.advance(blank, (self.keys(blank)[1], "call"), "3d")
        self.assertEqual(river.aggressor_history, (1, 1))
        self.assertEqual(choose_size(self.act(river, "check")), choose_size(replace(self.act(river, "check"), path=())))

    def test_public_abstraction_never_merges_different_sizing_textures(self):
        dry, wet = self.root(board="As Kd 7c"), self.root(board="9s 8d 6c")
        ranges = (parse_range("QhQc, JhJc"), parse_range("ThTc, 5h5c"))
        solver = Solver(dry, ranges, Settings(buckets=4, equity_samples=8))
        solver.bucket = lambda *args: 0
        hand = next(iter(ranges[0]))
        self.assertNotEqual(solver.info_key(dry, hand), solver.info_key(wet, hand))
        suit_rotated = replace(dry, board=tuple(c // 4 * 4 + (c % 4 + 1) % 4 for c in dry.board))
        self.assertEqual(choose_size(dry), choose_size(suit_rotated))
        self.assertEqual(board_features(parse_cards("As 3d 5c"))[0], "wet")

    def test_complete_trees_offer_jam_at_every_legal_decision(self):
        visited = 0
        for pot, stack in ((5, 97.5), (23, 88.5), (52, 74)):
            for board in ("As Kd 7c 2h 3d", "9h 8h 6c 7h 5h", "Ks Kd 4c 4h 2d"):
                runout = parse_cards(board)
                todo = [self.root(pot, stack, board=" ".join(board.split()[:3]))]
                while todo:
                    state = todo.pop()
                    visited += 1
                    self.assertAlmostEqual(state.pot + sum(state.stacks), 200)
                    self.assertGreaterEqual(min(state.stacks), 0)
                    if state.terminal:
                        self.assertFalse(state.actions(self.settings))
                        for winner in (-1, 0, 1):
                            self.assertAlmostEqual(state.payoff(winner, 0) + state.payoff(winner, 1), 0)
                        continue
                    if state.pending:
                        todo.append(state.deal(runout[len(state.board)]))
                        continue
                    choices = state.actions(self.settings)
                    aggressive = [a for a in choices if a.kind in ("bet", "raise")]
                    self.assertLessEqual(len(aggressive), 2)
                    self.assertLessEqual(len(choices), 4)
                    self.assertEqual(len({a.key for a in choices}), len(choices))
                    self.assertLessEqual(state.aggressions, 3)
                    call = max(state.paid) - state.paid[state.actor]
                    cap = min(state.stacks[state.actor], state.stacks[1-state.actor] + call)
                    jams = [a for a in aggressive if abs(a.amount - cap) < .001]
                    self.assertEqual(len(jams), int(cap > call + .001))
                    for a in aggressive:
                        self.assertGreater(a.amount, call)
                        self.assertLessEqual(a.amount, cap + .001)
                        if a.amount < cap - .001:
                            self.assertGreaterEqual(a.target + .001, max(state.paid) + state.last_raise if call else 1)
                        if a.amount >= cap - .001:
                            self.assertEqual(self.keys(state.apply(a)), ["fold", "call"])
                    todo.extend(state.apply(a) for a in choices)
        self.assertGreater(visited, 1000)

    def test_app_ignores_legacy_multiple_sizes_and_exports_model(self):
        payload = {"preflop": "raise:2.5/call", "flop": "As Kd 7c"}
        a, _, meta = prepare(payload)
        b, _, _ = prepare({**payload, "settings": {"sizes": [[3], [3], [3]], "max_raises": 0, "include_all_in": False}})
        self.assertEqual(a.root.actions(a.settings), b.root.actions(b.settings))
        self.assertEqual(a.settings.sizes, ((), (), ()))
        self.assertEqual(meta["sizing"]["max_aggressive_actions_per_decision"], 2)
        self.assertEqual(meta["sizing"]["model"], "scenario-v2")
        self.assertTrue(meta["settings"]["include_all_in"])
        self.assertFalse(meta["sizing"]["board_independent"])
        self.assertNotIn("flop_sizing", meta)
        self.assertEqual(a.inspect([])["sizing"]["scenario"], "donk")


if __name__ == "__main__":
    unittest.main()
