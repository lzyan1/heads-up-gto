# Heads up — deterministic early streets, direct rivers

A local study app for 100 BB heads-up no-limit hold’em, 0.5/1 blinds and no rake.
Preflop charts, flop policies and turn policies choose exactly one action for
an exact private hand at a decision. River decisions use a direct mixed-strategy
Nash solver. Different hands can take different actions, so range-wide action
totals and groups of suit combinations can still contain several colors.

This is a research prototype, not a certified full-game GTO policy. Deterministic
flop/turn policies restrict the game, and the local river solves do not provide
safe subgame-solving guarantees for the entire hand.

## Install and run

Requires Python 3.11+, NumPy, PyTorch and SciPy. Install all dependencies with:

```sh
python3 -m venv .venv                 # if needed
.venv/bin/python -m pip install -e .
.venv/bin/python run.py --open
```

Or double-click `run.command` on macOS after installing. The server binds to
http://127.0.0.1:8000 only. Use `--port 8001` for a different port. Training and
solving work offline after installation.

## Workflow

1. Follow the preflop action strip and choose a flop. The six GTO Wizard
   screenshot charts are converted to highest-probability actions. No manual
   postflop ranges are required.
2. **Solve continuation** trains the earlier-street networks. Current flop/turn
   action choices are deterministic. Each visited river betting boundary is
   solved directly, using ranges propagated through the earlier public actions
   and current deterministic policies. Private sampled opponent cards never
   determine the boundary strategy.
3. River continuation values integrate the compatible opponent range and use
   exact showdown payoffs. There are no river advantage or strategy training
   examples and no neural river-policy predictions.
4. After a complete training round, inspect earlier-street policies. The average
   action model publishes its highest-scoring action for each exact hand.
   Following actions filters the ranges automatically. On reaching the river,
   the app directly solves for those displayed-policy ranges; subsequent river
   decisions reuse that same boundary solution.
5. **Train more** continues earlier-street work. **Refine decision** starts a new
   local solve with the inherited ranges. Refining a river runs only the direct
   solver; refining a flop/turn uses the hybrid trainer.
6. Stop/time budgets interrupt an active training river worker. Resuming retains
   neural work and random states, but restarts the interrupted LP: HiGHS does not
   save a resumable optimizer checkpoint. Give a run enough time for its river
   solves to finish. A separate per-river LP limit defaults to 120 seconds.
7. Export includes root ranges/state, configuration and the displayed policy.
   River exports include a numerical equilibrium certificate for that boundary.
   There is no persistent training checkpoint; restarting discards live solves.

Settings affect new solves/refinements. Train more preserves existing model and
river settings. Only the two most recent study jobs are retained in memory.

An action with no reaching hands can produce an empty inspected branch. During
training, traverser deviations still need values even for zero-probability public
actions. When an entire range would become empty at such a deviation, the river
boundary retains the preceding range as its off-path belief. Counts are reported
as `river_off_path_solves` and disclosed in the UI. This convention is not a proof
of full-game equilibrium. Within a solved river, zero-reach private hands are
valued by best response to the fixed solved opponent policy, without revealing
the hand to that opponent or re-solving separately for each sampled deal.

## Preflop charts

`heads_up_gto/screenshot_charts.py` retains the original screenshot transcription
as source material, originally rounded to 10% steps. The active chart chooses
one highest-probability legal action per observed hand. Ties prefer fold, check,
call, then the smaller raise. Missing source cells remain unobserved.

Examples: AKo facing 11.5 always raises to 26 (previously 70% raise / 30% jam);
99 always calls (previously 50% call / 20% raise / 30% jam). No fractional chart
reach remains. Each player's chosen actions filter their range once along the
line. Board blockers and overlapping private cards are removed correctly.

Opening jams and BB 3-bet jams remain excluded. Some formerly mixed branches now
have no reaching hands; no replacement ranges are invented. The active chart
is downloadable at `/api/builtin-chart`, and its checked-in copy is
`charts/gto_wizard_screenshots.json`. It is not an equilibrium chart for the newly
restricted deterministic strategy space.

## Betting tree

The existing `scenario-v2` sizing rules remain: one contextual bet/raise size
plus all-in whenever legal. The size depends on pot type, position, public
aggressor history, board texture and effective stacks. Normal re-raises become
all-in. These are heuristic restrictions, not optimized GTO sizes. The same
betting engine and action menu are used by training, inspection and the LP.

## Solver implementation

- `deep_cfr.py`: deterministic greedy flop/turn policies, sampled advantage
  targets, bounded replay and neural fits. River requests yield to a reusable
  cancellable CPU worker instead of recursing through a river neural policy.
- `exact_river.py`: sequence-form linear programming with both players' private
  information preserved. Two HiGHS LPs find the strategies; recursive best
  responses independently check their deviation gains. “Exact” means the full
  specified river game solved to numerical tolerance, not symbolic arithmetic.
- `river_jobs.py`: a process reused across training boundaries, terminated on
  pause/cancel. Completed compact solutions have a bounded cache.
- `river_only.py`: direct river study/refinement without neural training.
- `range_game.py`: blocker-correct dealing, range conditioning and navigation.
- `game.py` / `sizing.py`: betting legality, payoffs and fixed action abstraction.
- `server.py` / `static/`: loopback API and browser study interface.

A failed/timed-out LP is an explicit error; it never substitutes a neural river
value. The certificate applies only to its fixed boundary ranges and allowed
river actions. Earlier-street predictions and sampling counts are not convergence
or exploitability certificates. Flop/turn models still use 128-wide networks,
25,000 samples per replay buffer, 128 traversals per player per round, and 256
optimizer steps per fit by default. Deterministic action selection changes the
standard mixed-policy Deep CFR algorithm, so its equilibrium claims do not carry
over unchanged.

## Validation and latency

```sh
.venv/bin/python -m unittest discover -s tests -q
node --test tests/test_ui.mjs
PYTHONPATH=. .venv/bin/python benchmarks/exact_river.py --include-full --output exact-river-results.json
PYTHONPATH=. .venv/bin/python benchmarks/exact_river.py --worker single_raised_dry 200 7 --lp-seconds 60
```

The exact tests embed standard Kuhn poker in Hold’em: board `2c 5d 7h 9s Jc`,
both ranges `As3s,Kh3h,Qd3d`, 2 BB pot and 1 BB stacks. Card blocking excludes
equal-rank deals, leaving six equally likely deals. The solver reproduces
−1/18 BB for the first player and the expected one-third bluff/call frequencies.
An independent exhaustive 64-by-64 pure-strategy payoff matrix confirms the
mixed equilibrium. Tests also verify per-hand continuation values, weighted
ranges, both players' best responses, deterministic earlier actions, boundary
integration, inspection consistency, and pause/resume behavior.

[Recorded random-range measurements](benchmarks/results/exact-river-m1-pro.json)
on an Apple M1 Pro with 16 GB RAM include construction, both LPs and independent
verification. With 100 hands per player, the three tested trees took about
0.063 s, 0.96–1.03 s and 1.61–1.64 s. At 200 hands, the larger dry tree took
17.7–21.4 s with an extended budget. All 1,081 legal hands per player took 12.3 s
in the smallest tree; the larger two hit a 10-second per-player LP cap before
certification. These are prototype measurements, not latency guarantees for
every training boundary. Imports/process startup are excluded from solver
latencies; worker reuse avoids repeated startup within a training run.

The LP approach follows the sequence-form formulation described in
[Zhang and Sandholm (2020)](https://proceedings.mlr.press/v119/zhang20x/zhang20x.pdf),
implemented with [SciPy/HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
