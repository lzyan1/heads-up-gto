import unittest
from dataclasses import replace
from heads_up_gto.cards import parse_cards, parse_range
from heads_up_gto.game import Settings, State
from heads_up_gto.server import prepare
from heads_up_gto.sizing import choose_sizes, opening_scenario, profile_for
from heads_up_gto.range_game import RangeGame


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

    def test_flop_uses_33_percent_in_every_pot_type_without_extra_jam(self):
        for pot, stack, target in ((5, 97.5, 1.7), (23, 88.5, 7.6), (52, 74, 17.2)):
            for board in ("As Kd 7c", "9h 8h 6c", "Ks Kd 4c", "Ah 8h 3h"):
                state = self.root(pot, stack, board)
                for node in (state, self.act(state, "check")):
                    with self.subTest(pot=pot, board=board, actor=node.actor):
                        self.assertEqual(self.keys(node), ["check", f"bet:{target:g}"])

    def test_barrels_donks_probes_and_delayed_cbets_use_distinct_turn_sizes(self):
        turn = self.advance(self.root(), ("check", "bet:1.7", "call"), "2h")
        self.assertEqual(turn.aggressor_history, (1,))
        self.assertEqual(opening_scenario(turn), "donk")
        self.assertEqual(self.keys(turn), ["check", "bet:4.2"])
        barrel = self.act(turn, "check")
        self.assertEqual(opening_scenario(barrel), "barrel")
        self.assertEqual(self.keys(barrel), ["check", "bet:5.5"])
        checked = self.advance(self.root(), ("check", "check"), "2h")
        self.assertEqual(opening_scenario(checked), "probe")
        self.assertEqual(self.keys(checked), ["check", "bet:2.5"])
        delayed = self.act(checked, "check")
        self.assertEqual(opening_scenario(delayed), "delayed_cbet")
        self.assertEqual(self.keys(delayed), ["check", "bet:2.5"])
        oop = self.advance(self.root(23, 88.5), ("bet:7.6", "call"), "2h")
        self.assertEqual(opening_scenario(oop), "barrel")
        self.assertEqual(self.keys(oop), ["check", "bet:25.2"])
        stab = self.act(oop, "check")
        self.assertEqual(opening_scenario(stab), "stab")
        self.assertEqual(self.keys(stab), ["check", "bet:19.1"])

    def test_river_barrels_and_probes_offer_two_sizes_but_donks_only_half_pot(self):
        turn = self.advance(self.root(), ("check", "bet:1.7", "call"), "2h")
        river = self.advance(turn, ("check", "bet:5.5", "call"), "3d")
        self.assertEqual(opening_scenario(river), "donk")
        self.assertEqual(self.keys(river), ["check", "bet:9.7"])
        barrel = self.act(river, "check")
        self.assertEqual(opening_scenario(barrel), "barrel")
        self.assertEqual(self.keys(barrel), ["check", "bet:6.4", "bet:19.4"])
        checked = self.advance(turn, ("check", "check"), "3d")
        self.assertEqual(opening_scenario(checked), "probe")
        self.assertEqual(self.keys(checked), ["check", "bet:2.8", "bet:8.4"])
        delayed = self.act(checked, "check")
        self.assertEqual(opening_scenario(delayed), "delayed_cbet")
        self.assertEqual(self.keys(delayed), ["check", "bet:2.8", "bet:8.4"])
        oop = replace(river, aggressor_history=(0, 0))
        self.assertEqual(self.keys(oop), ["check", "bet:6.4", "bet:19.4"])
        stab = self.act(oop, "check")
        self.assertEqual(opening_scenario(stab), "stab")
        self.assertEqual(self.keys(stab), ["check", "bet:6.4", "bet:19.4"])

    def test_raise_to_multipliers_and_single_raised_flop_reraise_pruning(self):
        for first in (("check", "bet:1.7"), ("bet:1.7",)):
            state = self.root()
            for key in first:
                state = self.act(state, key)
            self.assertEqual(self.keys(state), ["fold", "call", "raise:5.1"])
            raised = self.act(state, "raise:5.1")
            self.assertEqual(self.keys(raised), ["fold", "call"])
            self.assertEqual(choose_sizes(raised).scenario, "reraise_pruned")
            self.assertEqual(choose_sizes(replace(raised, path=())), choose_sizes(raised))
            self.assertTrue(self.act(raised, "call").pending)
        turn = self.advance(self.root(), ("check", "check"), "2h")
        facing = self.act(turn, "bet:2.5")
        self.assertEqual(self.keys(facing), ["fold", "call", "raise:6.2"])
        raised = self.act(facing, "raise:6.2")
        self.assertEqual(self.keys(raised), ["fold", "call", "raise:97.5"])
        river = self.advance(turn, ("check", "check"), "3d")
        for key in ("bet:1.7", "bet:5"):
            facing = self.act(river, key)
            self.assertEqual(self.keys(facing), ["fold", "call", "raise:97.5"])

    def test_three_bet_flop_reraise_and_turn_raise_shove(self):
        flop = self.root(23, 88.5)
        raised = self.act(self.act(flop, "bet:7.6"), "raise:22.8")
        self.assertEqual(self.keys(raised), ["fold", "call", "raise:88.5"])
        shoved = self.act(raised, "raise:88.5")
        self.assertEqual(self.keys(shoved), ["fold", "call"])
        self.assertEqual(self.act(shoved, "call").terminal, "showdown")
        turn = self.advance(flop, ("bet:7.6", "call"), "2h")
        facing = self.act(turn, "bet:25.2")
        self.assertEqual(self.keys(facing), ["fold", "call", "raise:80.9"])

    def test_four_bet_pot_shoves_after_called_flop_bet(self):
        flop = self.root(52, 74)
        facing = self.act(flop, "bet:17.2")
        self.assertEqual(self.keys(facing), ["fold", "call", "raise:74"])
        for line in (("bet:17.2", "call"), ("check", "bet:17.2", "call")):
            turn = self.advance(flop, line, "2h")
            for node in (turn, self.act(turn, "check")):
                self.assertEqual(self.keys(node), ["check", "bet:56.8"])
            river = self.advance(turn, ("check", "check"), "3d")
            for node in (river, self.act(river, "check")):
                self.assertEqual(self.keys(node), ["check", "bet:56.8"])

    def test_shove_boundary_is_strict_and_duplicate_river_sizes_merge(self):
        # 33 into 100 leaves 83, exactly half the 166 pot after a call.
        root = replace(self.root(), base_pot=100, start_stacks=(116, 116),
                       board=parse_cards("As Kd 7c 2h 3d"), street=2,
                       actor=1, checked=True, aggressor_history=(1, 1))
        self.assertEqual(self.keys(root), ["check", "bet:33", "bet:116"])
        below = replace(root, start_stacks=(115.99, 115.99))
        self.assertEqual(self.keys(below), ["check", "bet:115.99"])
        above = replace(root, start_stacks=(116.01, 116.01))
        self.assertEqual(self.keys(above), ["check", "bet:33", "bet:116.01"])
        # Facing 50 into 100: raise-to 125 leaves 175 of 300, exactly half
        # the called pot (100 + 125 + 125). Count only the additional call.
        turn = replace(self.root(), base_pot=100, start_stacks=(300, 300),
                       board=parse_cards("As Kd 7c 2h"), street=1, aggressor_history=(-1,))
        facing = self.act(turn, "bet:50")
        self.assertEqual(self.keys(facing), ["fold", "call", "raise:125"])
        short = replace(turn, start_stacks=(299.99, 299.99))
        self.assertEqual(self.keys(self.act(short, "bet:50")), ["fold", "call", "raise:299.99"])

    def test_effective_stacks_minimum_bets_and_short_all_ins(self):
        root = replace(self.root(), base_pot=100, start_stacks=(115.99, 1000),
                       board=parse_cards("As Kd 7c 2h 3d"), street=2,
                       actor=1, checked=True, aggressor_history=(1, 1))
        self.assertEqual(self.keys(root), ["check", "bet:115.99"])
        short = replace(self.root(52, 74), start_stacks=(.35, 1.))
        self.assertEqual(self.keys(short), ["check", "bet:0.35"])
        self.assertEqual(self.keys(self.act(short, "bet:0.35")), ["fold", "call"])
        tiny_pot = replace(self.root(), base_pot=1)
        self.assertEqual(self.keys(tiny_pot), ["check", "bet:1"])
        self.assertEqual(self.keys(self.act(tiny_pot, "bet:1")), ["fold", "call", "raise:3"])

    def test_board_cards_do_not_change_sizes_and_refinement_keeps_context(self):
        pending = self.act(self.act(self.act(self.root(board="9h 8h 6c"), "check"), "bet:1.7"), "call")
        blank = self.act(pending.deal(parse_cards("2s")[0]), "check")
        flush = self.act(pending.deal(parse_cards("2h")[0]), "check")
        self.assertEqual(choose_sizes(blank), choose_sizes(flush))
        for state in (blank, flush):
            refined = replace(state, path=())
            self.assertEqual(refined.actions(self.settings), state.actions(self.settings))
            self.assertEqual(choose_sizes(refined), choose_sizes(state))
        ranges = (parse_range("QhQc, JhJc"), parse_range("ThTc, 5h5c"))
        solver = RangeGame(blank, ranges, self.settings)
        hand = next(iter(ranges[0]))
        self.assertNotEqual(solver.info_key(blank, hand), solver.info_key(flush, hand))

    def test_complete_trees_preserve_legality_and_action_limits(self):
        streets, pruned, two_sizes = set(), 0, 0
        for pot, stack in ((5, 97.5), (23, 88.5), (52, 74)):
            for board in ("As Kd 7c 2h 3d", "9h 8h 6c 7h 5h", "Ks Kd 4c 4h 2d"):
                runout = parse_cards(board)
                todo = [self.root(pot, stack, board=" ".join(board.split()[:3]))]
                while todo:
                    state = todo.pop()
                    streets.add((state.sizing_profile, state.street))
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
                    self.assertLessEqual(len(aggressive), 2 if state.street == 2 else 1)
                    self.assertLessEqual(len(choices), 3)
                    self.assertEqual(len({a.key for a in choices}), len(choices))
                    self.assertLessEqual(state.aggressions, 3)
                    call = max(state.paid) - state.paid[state.actor]
                    cap = min(state.stacks[state.actor], state.stacks[1 - state.actor] + call)
                    if state.street == 0 and state.sizing_profile == "single_raised" and state.aggressions == 2:
                        self.assertEqual(self.keys(state), ["fold", "call"])
                        pruned += 1
                    two_sizes += len(aggressive) == 2
                    for a in aggressive:
                        self.assertGreater(a.amount, call)
                        self.assertLessEqual(a.amount, cap + .001)
                        if a.amount < cap - .001:
                            self.assertGreaterEqual(a.target + .001, max(state.paid) + state.last_raise if call else 1)
                            called_pot = state.pot + a.amount + a.target - state.paid[1 - state.actor]
                            self.assertGreaterEqual(cap - a.amount + .001, called_pot / 2)
                        else:
                            self.assertEqual(self.keys(state.apply(a)), ["fold", "call"])
                    todo.extend(state.apply(a) for a in choices)
        self.assertEqual(len(streets), 9)
        self.assertGreater(pruned, 0)
        self.assertGreater(two_sizes, 0)

    def test_app_ignores_legacy_sizing_inputs_and_exports_new_model(self):
        payload = {"preflop": "raise:2.5/call", "flop": "As Kd 7c"}
        a, _, meta = prepare(payload)
        b, _, _ = prepare({**payload, "settings": {"sizes": [[3], [3], [3]], "raise_multiplier": 4,
                                                  "max_raises": 0, "include_all_in": True}})
        self.assertEqual(a.root.actions(a.settings), b.root.actions(b.settings))
        self.assertEqual(a.settings.sizes, ((), (), ()))
        self.assertEqual(meta["sizing"]["max_aggressive_actions_per_decision"], 2)
        self.assertEqual(meta["sizing"]["model"], "scenario-v3")
        self.assertFalse(meta["settings"]["include_all_in"])
        self.assertTrue(meta["sizing"]["board_independent"])
        self.assertEqual(a.inspect([])["sizing"]["scenario"], "donk")
        self.assertEqual(a.inspect([])["sizing"]["targets"], (1.7,))


if __name__ == "__main__":
    unittest.main()
