"""CFR+ river solver against Kuhn-in-Hold'em and the exact LP oracle."""
import importlib.util
import random
import unittest

from heads_up_gto.cards import COMBOS, parse_cards, remove_blocked
from heads_up_gto.cfr_plus_river import CFRPlusRiver
from heads_up_gto.game import Settings, State
from tests.test_exact_river import kuhn_game

HAS_SCIPY = importlib.util.find_spec('scipy') is not None


class CFRPlusRiverTests(unittest.TestCase):
    def test_prefix_sum_showdown_matches_brute_force(self):
        import numpy as np
        from heads_up_gto.cards import score
        root = State(21, (89.5, 89.5), parse_cards('As Kd 7c 2h 3d'), street=2)
        pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
        rng = random.Random(11)
        shared = rng.sample(pool, 40)  # identical combos in both ranges
        ranges = [{h: rng.uniform(.1, 1.) for h in shared + rng.sample(pool, 120)} for _ in (0, 1)]
        solver = CFRPlusRiver(root, ranges, Settings())
        for player in (0, 1):
            reach = np.random.default_rng(player).random(solver.sizes[1 - player])
            got = solver._showdown(player, reach)
            for i, mine in enumerate(solver.hands[player]):
                want = np.zeros(4)
                for j, theirs in enumerate(solver.hands[1 - player]):
                    if set(mine) & set(theirs):
                        continue
                    a, b = score(mine, root.board), score(theirs, root.board)
                    want[0 if a > b else 1 if a < b else 2] += reach[j]
                    want[3] += reach[j]
                np.testing.assert_allclose(got[:, i], want, atol=1e-12)

    def test_cfr_plus_variant_converges(self):
        solver = CFRPlusRiver(*kuhn_game(), algorithm='cfr+').solve(iterations=2000)
        self.assertAlmostEqual(solver.stats['value_bb'], -1 / 18, places=3)

    def test_kuhn_value_and_exploitability(self):
        solver = CFRPlusRiver(*kuhn_game()).solve(iterations=2000)
        self.assertAlmostEqual(solver.stats['value_bb'], -1 / 18, places=3)
        self.assertLess(solver.stats['exploitability_bb'], 1e-3)

    @unittest.skipUnless(HAS_SCIPY, 'SciPy needed for the LP oracle')
    def test_matches_exact_lp_on_random_ranges(self):
        from heads_up_gto.exact_river import ExactRiver
        root = State(21, (89.5, 89.5), parse_cards('As Kd 7c 2h 3d'), street=2,
                     sizing_profile='single_raised', aggressor_history=(1, 1))
        pool = list(remove_blocked(dict.fromkeys(COMBOS, 1.), root.board))
        rng = random.Random(3)
        ranges = [{h: rng.uniform(.05, 1.) for h in rng.sample(pool, 25)} for _ in (0, 1)]
        exact = ExactRiver(root, ranges, Settings()).solve(tolerance=1e-7)
        approx = CFRPlusRiver(root, ranges, Settings()).solve(iterations=3000)
        self.assertLess(approx.stats['exploitability_bb'], 0.02)
        self.assertAlmostEqual(approx.stats['value_bb'], exact.stats['value_bb'], delta=0.05)


if __name__ == '__main__':
    unittest.main()
