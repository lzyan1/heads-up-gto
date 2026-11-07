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

The `scenario-v3` table offers one aggressive action per spot, except for the
two non-donk river bet sizes. Board texture does not change the menu:

| Spot | Flop | Turn | River |
| --- | --- | --- | --- |
| Aggressor c-bet / barrel | 33% | 66% | 33% or 100% |
| Previous street checked through: delayed c-bet / probe | n/a | 50% | 33% or 100% |
| Non-aggressor bets after the aggressor checks | 33% | 50% | 33% or 100% |
| Donk lead into the previous aggressor | 33% | 50% | 50% |
| Raise facing a bet / donk | 3x | 2.5x | All-in |
| Re-raise facing a raise | All-in, except pruned in single-raised pots | All-in | n/a |

Bets are percentages of the pot **before** betting. Raises are **raise-to**
multiples of the opponent's street-total bet, with legal minimum raises applied.
All-in is not an extra action: it replaces a prescribed bet or raise when the
effective stack is too short, or the remaining effective stack would be
**less than half the pot after the opponent calls**. Equality does not trigger
this shove rule. Normal amounts are rounded to 0.1 BB, exact short all-ins and
minimum raises are preserved, and duplicate river targets are merged.

These are fixed modeling restrictions, not optimized GTO sizes. The same
betting engine and action menu are used by training, inspection and the LP.
Preflop chart sizes are unchanged. Previously recorded benchmarks used the
older tree and do not measure this sizing model.

## Offline river dataset

Generate supervised river-value examples independently of the app and neural
training. Run from the project directory (on macOS, `caffeinate -i` prevents
idle sleep while the script runs):

```bash
caffeinate -i .venv/bin/python scripts/precompute_rivers.py --hours 8 --output data/river-dataset
```

The same command resumes the same dataset. `--hours` is a fresh time budget
for each invocation. For a small first run, add `--samples 20`: this is a target
**total** count including previous runs, not 20 additional samples. Alternatively
run `.venv/bin/python -m heads_up_gto.river_dataset` with the same arguments.
No extra dependencies beyond the project's existing NumPy/SciPy are needed.

Each completed block of 20 solved spots contains **9 single-raised, 9 3-bet,
and 2 4-bet** pots (45/45/10). Formation order is randomized. Folds, pre-river
all-ins, and failed LPs do not consume those quotas. No 5-bet pots are used:
the chart's 5-bet is all-in and has no postflop decision.

Two deliberately naive bots independently sample fixed value/bluff thresholds
for each hand. They rank made hands, pair strength, overcards, gutshots,
open-ended straight draws and flush/combo draws. Calling uses the value
threshold with a simple adjustment for the price. Decisions are deterministic
for a hand and public state; no equities, opponent hole cards or future cards
enter their policies. After each action, the entire acting player's range is
filtered by that same rule. Cards are sampled with the correct private-card
blockers; complete surviving ranges are retained, never narrowed for speed.
The bots stop before the first river action, where the existing LP takes over.

Output files:

- `manifest.json`: seed, formation mix, bot/sizing versions, solver limits and
  source/chart fingerprints. A changed configuration requires a new directory.
- `events.jsonl`: authoritative results and LP failures, flushed and synced
  after every solve. Interrupted attempts are retried; incomplete final journal
  lines are recovered automatically. A complete corrupt line causes an error.
- `spots-000001.csv`, etc.: one row per spot, **2,500 spots per file** by default
  (`--chunk-size`). A partial final file is also finalized on exit. An active
  shard uses `.csv.part`; CSVs are rebuilt from the journal on resume.
- `progress.json`: saved count, formation counts, next attempt and LP failures.

Nested CSV cells are JSON: decode them with `json.loads`. Rows include the full
river state/history, both exact-combo ranges, root mixed strategy, and conditional
per-hand equilibrium EVs for **both players**, in BB and fractions of the river
pot. Player order is BB/OOP then SB/IP. A hand with no compatible opponent has
a `null` EV, which must be masked when training; it is not a zero-value label.
Zero-weight hands outside the supplied ranges are not labeled. Bot thresholds
and sampled private hands are audit metadata, **not value-network inputs**.

One reusable CPU process solves spots sequentially to limit laptop memory use.
Defaults: 120 seconds for the LP pair, a 180-second total per-spot watchdog,
and a numerical certificate tolerance of 1e-7 BB. `--lp-seconds`,
`--spot-seconds`, `--tolerance`, `--seed`, and `--max-payoff-entries` are
configurable; use the same settings when resuming. Broad ranges can take
minutes per spot. Timeouts are logged, never converted into training labels;
check their frequency because they can bias which spots enter the dataset.
Progress is printed every 30 seconds and after each solve. Ctrl-C or SIGTERM
stops the active worker, preserves completed examples, and finalizes the CSV.
Only one process may write a particular dataset directory at a time.

This script generates data only. It does not fit a network or alter Deep CFR.

## Full-width postflop DCFR (experimental, `cfr+` branch)

`heads_up_gto/postflop_cfr.py` (`PostflopCFR`) solves a flop, turn or river root over the whole
remaining game with no neural networks and no sampling. Every hand is an array element and every
dealt card is a leading "context" axis, so one traversal updates all hands in all runouts.
All-ins before the river deal every remaining card set. The algorithm is the DCFR variant used
by b-inary/postflop-solver (`algorithm='cfr+'` is also available).

```python
from heads_up_gto.postflop_cfr import PostflopCFR
solver = PostflopCFR(root_state, (oop_range, ip_range), Settings(...))   # raises if too big
solver.solve(iterations=1000, target_exploitability=0.001 * root_state.pot)
solver.strategy(['check', 'bet:10.5', 'call', '9s'])   # (actions, [actions, hands] probabilities)
```

Two engines share the same tree and tests. `engine='numpy'` is the readable reference
(float64). `engine='numba'` (the default when `pip install -e '.[fast]'` is done) flattens the tree
into integer tables and runs each runout's subtree in compiled, threaded kernels on cache-resident
vectors, in float32 by default. It is about 30x faster than the numpy engine (the first call
compiles for several seconds, cached afterwards), within roughly 2x of postflop-solver, and agrees
with the numpy engine to rounding on river, turn and flop roots. It uses the performance cores
(`threads=` overrides).

Memory is the practical limit, because regrets and strategies are stored for every river runout
(`solver.memory_bytes`, guarded by `max_bytes`). Turn roots are comfortable; flop roots need
narrow ranges or a small betting tree. `benchmarks/postflop_full_compare.py` checks values and
latency against postflop-solver (build `benchmarks/postflop_cli` with cargo first).

### In the web app

Settings → **Solver** chooses the backend per solve (default **Full tree**; the old Deep CFR +
river-LP flow is still available as **Legacy**, and `/api/solve` takes `"solver": "full_tree" | "legacy"`).
With Full tree every street can mix, the status line shows exploitability as a percentage of the pot,
Stop pauses after the current iteration and Train more continues from where it stopped. Typical
cost on the built-in charts: river and turn solves reach 0.05% of the pot in seconds; a 3-bet flop
takes minutes; a single-raised flop needs about 9 GiB and roughly 5-8 s per iteration, so give it a
long time limit. A solve whose memory estimate exceeds 60% of physical RAM is rejected up front, and
older solves are dropped from memory to make room for a new one.

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
