# Heads up — postflop study

A local Python application for 100 BB heads-up no-limit hold'em: use the hardcoded GTO Wizard preflop charts, select a flop, and explore the remaining betting decisions. Displays a 13×13 range grid, exact-combo selection, mixed action frequencies, and per-decision training counts.

**This is an experimental approximate solver, not a certified GTO policy.** It implements the chart-conditioned postflop part of the proposed architecture. Preflop frequencies remain fixed. It does not implement a full-game preflop optimization loop or safe subgame solving.

## Run

Requires Python 3.11+. The tabular backend has no runtime dependencies and works offline.
Deep CFR is optional; install its dependencies once with `.venv/bin/python -m pip install '.[deep]'`.

```sh
.venv/bin/python run.py --open
```

Or double-click `run.command` on macOS. The interface is at http://127.0.0.1:8000. Stop the server with Control-C. Choose another port with `--port 8001`. The server binds only to loopback and is not intended for hosting.

## Workflow

1. The workspace opens at the SB's first preflop decision, using your six GTO Wizard decision charts rounded to 10% steps. Pick an action in the **horizontal history strip** or the action cards on the right. Each choice adds the next player's decision, with remaining stacks and pot size.
2. When a call or check closes preflop, a **card-picker popup** opens. Select three flop cards from the four suit rows and choose **Solve continuation**. Clear, shuffle, and deselect controls are included. Closing the popup leaves a “Choose flop” tile so you can return later.
3. Follow postflop actions using the same strip. When betting ends, choose a turn or river card in the popup. Cards already on the board are disabled. A fold, called all-in, or completed river ends the line without asking for another betting street.
4. **Hover over the 13×13 grid** to see every suit combination and its action frequencies on the right. Click a hand to pin it, click again or use Unpin to release it. Keyboard focus and arrow keys also inspect hands. Blocked, absent, and untrained combinations are labeled explicitly.
5. Click a prior decision's **SB/BB heading** to inspect it while keeping the later line visible. Selecting a different action replaces the later branch. Click a board tile to edit it; changing a turn/river discards subsequent actions, and changing the flop starts a new solve.
6. **Strategy** colors each hand by its conditional action mix. **Ranges** shows reach by bar height. **Breakdown** combines reach height with action colors, like the supplied screenshots. Green means check/call, blue means fold, and reds mean bets/raises, with dark red for all-in.
7. **Settings** uses the hardcoded GTO Wizard charts exclusively. Chart import, demo selection, and manual ranges have been removed from the app. All streets offer one scenario-based bet or raise size plus an all-in whenever legal; sizes change with the pot type, position, betting history, board and effective stacks. Settings offers an iteration count or a time limit (seconds), plus card model, strength buckets, equity samples, random seed and an optional decision limit. Load a specific spot and Download charts are also available; custom bet-sizing inputs stay removed. It remains available during training; the header stays visible when scrolling.
8. **Train more** continues the same solver with a fresh iteration or time budget, preserving its card model and seed. A changed optional decision limit takes effect on the resumed run. **Refine decision** starts a local solve at the displayed board/action state using conditioned ranges and the saved settings. Earlier actions stay in the strip; the previous solve remains navigable while retained in memory. Refinement is useful when deeper branches are untrained. It is not a safe re-solve and has no full-game exploitability guarantee. Uniform propagation through untrained earlier decisions is explicitly disclosed.
9. **Export** saves the displayed decision as JSON, including configuration, root state, root ranges, training statistics, and combo probabilities. Live solver state is in memory; restarting the app discards it. Only the two most recent solves are retained. Older decisions in a refined history are disabled once their solve expires.

The cards use `s` ♠, `h` ♥, `d` ♦, and `c` ♣. All recommendations are for the **player currently to act**, not a fixed hero seat. Raises are total amounts in BB. For example, `raise:2.5 / raise:11.5 / call` means SB raises to 2.5, BB raises to 11.5, and SB calls.

For tabular training, coverage means the fraction of compatible range mass with both regret updates and average-strategy samples, not convergence. Combo tooltips include training counts. Missing chart frequencies and untrained tabular decisions have no invented recommendation. Tabular action averages exclude unknown mass. Deep CFR displays model predictions separately from directly sampled coverage; its action averages include predicted unseen hands. Neither backend is a certified GTO solver.

## Postflop action abstraction

The app uses `heads_up_gto/sizing.py` (`scenario-v2`) on **all three streets**. Each node offers check plus the scenario bet and an all-in, or fold/call plus the scenario raise and an all-in. Facing an all-in leaves fold/call. When the scenario size is already all-in, it appears only once. The old screenshot flop trees and configurable multi-size turn/river controls are removed. The preflop screenshot charts remain unchanged.

These are deterministic **heuristics, not optimized GTO sizes**. No GTO Wizard postflop ranges or frequencies are imported. The solver chooses action frequencies within this restricted game. A smaller tree makes training cheaper; it does not establish convergence or full-game optimality.

Flop opening sizes are pot fractions, selected separately for each situation:

| Pot / acting role | Dry | Wet | Paired | Three-suited |
| --- | ---: | ---: | ---: | ---: |
| Single-raised · SB continuation bet | 33% | 67% | 25% | 25% |
| Single-raised · BB donk | 75% | 100% | 50% | 50% |
| 3-bet · BB continuation bet / SB stab after a check | 33% | 50% | 25% | 25% |
| 4-bet · SB continuation bet | 25% | 33% | 20% | 20% |
| 4-bet · BB donk | 50% | 67% | 33% | 33% |

“Wet” means an unpaired two-suited board or three ranks within a five-rank straight window (including the wheel). Three-suited takes precedence over paired, then wet, then dry. These coarse categories do not estimate range or nut advantage. The larger flop donks are a requested simplification, **not a claim that solved donks are generally large**.

| Later-street situation | Turn base | River base |
| --- | ---: | ---: |
| Previous-street aggressor barrels | 75% | 100% |
| BB probes after checks | 75% | 75% |
| Preflop aggressor makes a delayed bet | 67% | 75% |
| SB bets after the prior aggressor checks | 50% | 67% |
| BB leads into the prior-street aggressor | 33% | 33% |

Wet-board probes/delayed bets/stabs use 100%/75%/67%. River barrels on wet boards use 75%; on dry boards a third consecutive barrel uses 125%. Four-bet turn bets cap at 50%. Paired boards, three-flush boards, or a newly pairing/draw-completing card cap the choice at 50%; four-flush or four-to-a-straight boards cap it at 33%. Only dealt cards and public actions select the size, never the player's private hand. The solver's public abstraction records these texture features so incompatible sizes cannot share an information set.

A normal raise adds a fraction of the **pot after calling** to the outstanding street total: flop 75% OOP / 50% IP (50% in 4-bet pots), turn 75% / 67%, river 100% / 75%. Enforce the minimum legal full raise. After that normal raise, the only re-raise size is all-in; this limits a street to bet → raise → jam. Calling at any point can close the street. This intentionally removes smaller re-raises.

Use a jam in place of the selected amount when an opening stack-to-pot ratio is at most 0.75, the amount commits at least 75% of the effective stack, or it would leave at most 20% of the pot after a call. Normal amounts round to 0.1 BB, with a 1 BB minimum opening bet; legal minimum raises and exact short all-ins override rounding. The effective stack caps every action. A separate effective-stack all-in is always available when legal, including when checking, donking, betting or raising is possible.

The current decision explains the sizing rule in the UI. Exports include the model version, profile, sizing tables, current rule and public aggressor history. Local refinement preserves that history even when resetting its solver path. Legacy API sizing controls are normalized and cannot change this menu or disable the all-in; the generic engine remains available only to internal diagnostic games. Start a new solve after updating.

Design references: [GTO Wizard's conventional size examples](https://blog.gtowizard.com/all-you-need-to-know-about-our-solutions/) and [custom tree documentation](https://help.gtowizard.com/how-to-build-custom-solutions/) inform the terminology and use of SPR. The numeric tables above are our modeling choices, not a transcription or endorsement from those references.

## Built-in screenshot charts

The six supplied GTO Wizard decision charts are the app’s only preflop source. They are hardcoded in `heads_up_gto/screenshot_charts.py`; `charts/gto_wizard_screenshots.json` is an equivalent readable export. No images or image libraries are needed to run the app. The `/api/builtin-chart` endpoint exports the current built-in chart.

| Image | Player's decision | Actions |
| --- | --- | --- |
| 1 · 19:26:33 | SB opening | Fold / raise to 2.5 |
| 2 · 19:26:40 | BB facing 2.5 | Fold / call / raise to 11.5 |
| 3 · 19:27:04 | SB facing 11.5 | Fold / call / raise to 26 / all-in to 100 |
| 4 · 19:27:11 | BB facing 26 | Fold / call / all-in to 100 |
| 5 · 19:27:20, confirmed Oct 19 at 13:03:03 | SB facing 100 after raising to 26 | Fold / call |
| 6 · Oct 19 at 13:02:54 | BB facing SB’s 4-bet all-in | Fold / call |

**Preflop jam restriction:** SB’s 4-bet jam after 2.5 → 11.5 and BB’s 5-bet jam after 2.5 → 11.5 → 26 are available. Opening jams and BB 3-bet jams are disabled. The original SB shove mixes are restored: AKo uses 30% jam / 70% raise to 26; 99 uses 30% jam / 20% raise / 50% call. Both shove responses have only fold/call; a call ends the hand at 200 BB, without opening a flop solve. The Oct 19 SB response screenshot confirms the earlier chart. The new BB response includes approximate calls of ATs 20%, KQs 80%, KJs 10%, AQo 90% and 55 90%; short pure-green bars such as 77 and 66 remain 100% calls conditional on reaching the decision. These are rounded chart transcriptions, not a fresh equilibrium solve.

These are **visual approximations, not an exact GTO Wizard export**. Green is call, blue is fold, bright red is the smaller raise, and dark red is the 100 BB all-in. Each visible conditional mix is rounded to 10% increments and totals 100%; near a boundary, neighboring rounded values can be reasonable. Black vertical space in the later Breakdown charts represents earlier reach and is excluded from the action proportions. A short green bar therefore means a pure call at that node. Its earlier reach is reconstructed by multiplying the preceding conditional frequencies exactly once.

Example: BB A8s raises to 11.5 at approximately 30% and then always calls the raise to 26. Its resulting 4-bet-pot weight is 30%, not 9%. Image 5's short TT bar similarly means always call the shove conditional on having raised to 26. The darkened KJs opening cell is interpreted as selection shading of Raise 2.5, rather than a jam: image 3 shows KJs fully present after the 2.5 raise.

Completely uncolored hands are recorded as unobserved, not fabricated pure folds. All hands with positive reach after rounding have a defined strategy at the next supplied node. The all-in response is retained in the chart even though an all-in preflop line has no postflop decisions. No limp or alternate-size branches are inferred. Unsupported lines fail explicitly; the app does not substitute other ranges.

The screenshot presets are single-raised (5 BB pot / 97.5 BB behind), three-bet (23 BB / 88.5 BB), and four-bet (52 BB / 74 BB).

## Chart storage

`charts/gto_wizard_screenshots.json` is the downloadable copy of the hardcoded strategy, including the preflop jam restriction. Node keys contain the preceding action history and each action maps to conditional hand frequencies. The app multiplies each player’s action frequencies along the line once, removes board-blocked combinations, and conditions on no overlap between the two players’ cards.

The server accepts only the hardcoded source. Legacy demo/import/manual API requests are rejected rather than silently ignored. Generic chart utilities and the synthetic example remain as internal test fixtures, not selectable application modes.

## Solver design and limits

- External-sampling Monte Carlo CFR: sample a legal private deal and a complete runout; enumerate the traverser's actions and sample the opponent's actions from its current strategy. Alternate updates for both players. At sampled opponent nodes, accumulate the average policy. Regret matching uses positive cumulative regrets; negative regrets are retained (vanilla MCCFR, not CFR+).
- Zero-sum terminal utilities include half the initial pot plus the opponent's subsequent investment on a win, and minus half the initial pot minus the player's investment on a loss. Ties split the pot. Betting legality and chip conservation are tested.
- **Strength buckets (default):** deterministic sampled showdown equity against the input opponent range assigns each hand to one of 8 buckets. Later public boards are grouped by rank multiplicities, maximum suit count, presence of a new highest rank, and the texture features used by the sizing model. Information-set keys retain the whole history of private buckets, public observations, and actions. No hidden opponent card or future community card enters a strategy key. This defines a coarse perfect-recall observation game; it loses strategically important blocker and board information. It is intentionally a prototype abstraction, not a production poker abstraction.
- **Exact cards:** keys contain the actual hole cards and visible board. This removes card bucketing but retains the action abstraction and Monte Carlo training error. Broad three-street ranges require far more samples and memory. Choose Strength buckets or Exact cards in Settings. Exact mode is most useful for narrow ranges or refined river spots.
- All-in and showdown values use the actual sampled cards and exact hand evaluator in both modes. There is no heuristic terminal payoff or artificial reveal of the runout.
- Strength estimates use the fixed root opponent range, not an adaptively recomputed range at each future action. They are fixed features of the abstraction, not values used to replace terminal utilities.
- Iteration counts, node counts, and visited range mass are diagnostics, not exploitability bounds. Full three-street exploitability is not computed. `diagnostics.river_exploitability` evaluates exact best responses for small exact-card river games (at most 400 legal hand pairs), and is used in the convergence test.
- The optional decision limit is blank (unlimited) by default for new solves and refinements. A chosen limit stops with partial results; clear or increase it before Train more to continue. Physical RAM is finite; a catchable Python memory exhaustion retains partial results, but operating-system termination cannot be prevented. An interrupted traversal may leave partial updates.
- Training can stop after an iteration count or a wall-clock time limit. Timed training finishes both players’ traversals of the current iteration before stopping, so it can slightly exceed the requested duration. Preparation and displaying the range are outside the training timer. Every solve, refinement or resumed run gets a fresh budget; cumulative training time and iterations are retained. The API accepts either `iterations` (integer, at least 10) or `time_limit_seconds` (positive finite number), never both. Progress includes per-run `run_seconds`, cumulative `seconds`, and `time_limit_seconds`; a timed run has no iteration `target`. Stop remains available in either mode.
- Local re-solves condition on the existing policy; they do not preserve opponent counterfactual-value guarantees and can change earlier strategic incentives. Preflop chart quality, abstraction error, sampling error, and subgame consistency all remain material limitations.

The next research step toward the original full-game proposal would be a preflop optimization loop with range-dependent per-hand continuation values, principled safe subgame boundaries, and exploitability evaluation. This application provides the local engine, chart boundary, and study UI; it does not claim that final step is complete.

## Test

```sh
.venv/bin/python -m unittest discover -s tests -v
node --test tests/test_ui.mjs
```

Node is only needed for the browser-model tests; it is not a runtime dependency.

Tests cover 5–7 card evaluation including wheel/full-house edge cases; range expansion; chart multiplication; illegal preflop lines; scenario sizing on all streets, board-dependent menus, public information-set separation, and refinement continuity; minimum raises, all-ins and stack conservation; blocker-correct joint sampling; deterministic resumed training; no fabricated untrained recommendations; and measured exploitability improvement in a small exact river game.

## Code map

- `heads_up_gto/cards.py`: cards, ranges and evaluator.
- `heads_up_gto/preflop.py`: preflop action reconstruction and chart validation.
- `heads_up_gto/screenshot_charts.py`: six hardcoded user-supplied decision charts in 10% steps.
- `heads_up_gto/sizing.py`: scenario and board-dependent postflop sizing model, with an additional legal all-in.
- `heads_up_gto/navigation.py`: partial preflop decisions, chart-conditioned grids, and action history.
- `heads_up_gto/game.py`: immutable betting state and legal abstract actions.
- `heads_up_gto/solver.py`: MCCFR, card abstraction, range conditioning, inspection.
- `heads_up_gto/diagnostics.py`: small-river exact best-response diagnostic.
- `heads_up_gto/server.py`: loopback HTTP API and one active background solve.
- `heads_up_gto/static/`: responsive browser UI; no remote assets or scripts.

Research references: [Lanctot et al., Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html); [Brown and Sandholm, Safe and Nested Subgame Solving](https://arxiv.org/abs/1705.02955). Safe subgame solving is a reference for future work, not a feature currently implemented.

## Coverage diagnostics (exact-hand investigations)

The displayed percentage is the compatible conditional range mass with **both**
regret updates and average-policy samples. It is not a convergence percentage.
The decision panel reports unseen, regret-only, average-only, and low-sample hands;
individual combinations show both counters. The threshold of 200 of each is a
rough visibility aid, not a statistical confidence bound. Action totals still
exclude hands without a displayed policy, so sparse coverage can distort totals.

`POST /api/jobs/<id>/diagnostics` with `{}` returns stored-node sample distributions
by street after training stops. This excludes unseen information sets; use
`POST /api/jobs/<id>/node` to include unseen compatible hands at a selected path.
The latter includes `training.counts`, conditional `training.mass`, and min,
median, p90, max and total for regret updates and average samples. Under bucketed
mode, several hand rows can share the same underlying counters.

Training statistics distinguish requested target, completed iterations, attempted
iterations, completed traversals by player, interrupted iterations, tree visits,
and the node cap. `partial_updates_possible` means an interrupted attempt may
have left updates behind; it does not imply all attempts were rolled back.
Timed runs finish both traversals; stop, node-limit and memory failures can
interrupt them. Resuming repeatedly into the same node cap is not useful.

To reproduce the 130,000-iteration fixed-flop exact-hand experiment (can consume
substantial memory for millions of nodes), run from the project directory:

```sh
PYTHONPATH=. python3 benchmarks/coverage.py --output coverage-results.json
python3 -m unittest discover -s tests -q
node --test tests/test_ui.mjs
```

The diagnostic tests exercise the production `_walk` on Kuhn poker, compare it
against all pure best responses, and check exact river best responses with
all-ins enabled. These validate small games, not full Hold'em exploitability.
For focused turn study, use the existing Refine action after selecting a board.
This concentrates samples on that decision but relies on inherited approximate
ranges and is not safe full-game equilibrium refinement. Large bets and all-ins
remain available; no hand bucketing is needed for exact mode.

## Experimental enumeration of turn cards

In Settings, select **Board sampling → Enumerate every turn · experimental**.
This automatically selects exact cards. For the comparison spot, load flop
`As Td 6h` with preflop history `raise:2.5/raise:11.5/call` (23 BB pot,
88.5 BB remaining per player). Choose a time limit when comparing approaches:
one enumerated iteration does much more work than a sampled iteration.

API setting: `{"mode":"exact","chance_mode":"enumerate_turns"}`. The default
`chance_mode` remains `sampled`. Resuming preserves the original traversal mode;
start a new solve to compare modes. The existing all-in and large-bet options
are unchanged. New solve metadata and statistics include `chance_mode`;
`enumerated_turns` counts the runouts prepared across sampled hand pairs.

Each iteration still samples a compatible private-hand pair. It prepares all
45 legal turn cards and one uniformly sampled river from the 44 remaining
cards for each turn. Both traversals reuse this bank. When betting advances
from flop to turn, all turn branches are visited and their utilities are
averaged. Descendant regret and average-policy sums receive the corresponding
1/45 chance weight; visit counts still count actual visits. Flop all-in
showdowns also average over this bank. No unobserved river enters a turn key.
If the solve starts on the turn or river, normal runout sampling applies.
This is **turn enumeration**, not full game-tree or 1,980-runout enumeration:
opponent actions, private hands and rivers remain sampled.

The experiment can spend less time on distinct private hands and repeated
regret updates. Broader board coverage need not mean better convergence, and
raw iteration counts are not comparable. Node-budget interruption remains
possible and is disclosed by the diagnostics.

Reproduce the equal-time experiment from the project directory:

```sh
PYTHONPATH=. python3 benchmarks/turn_sampling.py --seconds 60 --seeds 7 23 --output turn-sampling-results.json
```

The benchmark uses a two-million-node guard per run (override with `--max-nodes`,
where 0 means unlimited). It records actual completed work, full decision rows,
coverage, per-street sample distributions and configuration. A capped run is
not an equal-time comparison. Small tests cover all 45 legal turns, blockers,
conditional showdown winners, 1/45 weighting, early all-in evaluation, resume,
and API export. Existing Kuhn and river convergence tests remain in the suite.

The observed GTO Wizard solution uses a 12.1 BB flop bet and no immediate shove
at its root. This app's three-bet-pot profile uses 7.6 BB plus an 88.5 BB shove
on this flop; preflop frequencies are rounded. Treat Wizard frequencies as a
qualitative reference, not an exact numerical convergence target for this tree.

## Shared-deal update-order fix

Both player traversals reuse a sampled deal. Their policies must therefore stay
fixed for that entire iteration. The solver now caches the pre-update strategy
at every visited information set and releases the cache after both traversals,
or on interruption. Regret and average accumulators can be updated in place,
but those changes do not affect actions until the next iteration.

Previously player 1's traversal could respond to a player 0 strategy just
updated from the same hidden hands/runout. This biases the sampling estimator,
particularly at sparsely visited information sets. A regression game with an
unobserved fair coin gives an exact first-iteration expected call-regret
increment of +0.25 under the uniform profile; the old ordering gives 0, and
frozen policies restore +0.25. Existing Kuhn and river tests alone did not catch
this early-iteration bias.

Stats label corrected runs with `policy_update: simultaneous_frozen`. Start a
fresh solve after restarting the server; old strategy accumulators were produced
by the previous update order. This fix does not certify full Hold'em convergence
or make this action menu equivalent to GTO Wizard's. All-in actions remain.

`benchmarks/policy_freeze.py` reproduces a short equal-time comparison against
the legacy update order. Its near-pure frequency counts compare current regret
matching with the displayed historical average; a more nearly pure current
strategy is not automatically a better strategy. Mixed averages alone do not
prove that a solver is wrong.

## Deep CFR pilot for a 16 GB Apple Silicon Mac

The optional `deep_cfr` backend learns across exact-card states instead of allocating
an independent regret table for every combination. It supports a **fixed flop,
turn or river root with fixed input ranges**. A flop-root solve trains all three
streets together, sampling private hands, opponent actions and future board cards.
It retains the existing fixed scenario sizes and all-in menu; it does not solve
unrestricted no-limit Hold'em or certify unseen predictions.
The tabular backend remains the default.

### Install and use

```sh
python3 -m venv .venv                 # only if this checkout has no environment
.venv/bin/python -m pip install '.[deep]'
.venv/bin/python run.py --open
```

1. In **Settings → Training method**, select **Deep CFR · flop / turn / river**.
   Selecting it switches the budget control to a time limit. Set your budget and
   save settings; this does not start training or convert an existing solve.
2. Start a new hand, follow preflop, and choose the flop. **Solve continuation**
   starts Deep CFR directly from that flop. Alternatively, enter the preflop line
   and board under **Load a specific spot**, then choose **Load spot & solve**.
3. Navigate to any turn/river afterward to inspect the same flop-trained average
   strategy. No additional solve is needed. **Refine decision** remains available
   for a separate local solve with inherited ranges, but it is not the original
   flop model. Poorly trained earlier ranges remain a source of error; uniform
   propagation is disclosed. Refinement is not safe subgame solving.
4. Use **Train more** to continue. Stop and timed budgets pause between traversal
   batches or optimizer steps, including in the middle of a round. The unfinished
   round resumes in place with its RNGs, buffers, models and optimizer intact.
   The first prediction appears after one complete round; later pauses keep the
   last completed average strategy available. The budget can overrun by the
   current batch/step, plus device initialization. It does not wait for a whole round.
5. The range grid shows **predicted average strategies**, including unseen hands.
   **Direct samples** is a separate compatible-range-mass measure requiring both
   advantage and strategy visits. Neither prediction availability nor visit counts
   measure accuracy. Once the 50,000-state visit tracker fills, counts/coverage are
   lower bounds. Replay sampling and neural learning continue normally.

A neural round includes `deep_traversals` deals for **each** player, two advantage
fits and two average-policy fits. Its count is not comparable to tabular iterations.
Compare quality at equal elapsed time. Changing neural settings requires a new
solve or refinement; Train more keeps the existing architecture and buffers. Models live
in memory, like the tabular solver: JSON export records the displayed strategy
and configuration, not a resumable neural checkpoint. Restarting loses training.

### Supplied GTO Wizard line

In **Settings → Load a specific spot**, choose **Set up supplied GTO Wizard flop**.
This fills in Deep CFR, SB raise 2.5 / BB call, and **As Kd 7c** without starting
training. Choose your time limit, then **Load spot & solve** when ready. The default
CPU, 128-wide networks and 25,000-entry buffers are unchanged. Longer flop rounds
may pause before producing a first prediction; **Train more** continues that round.

After stopping or finishing, **Compare line** inspects seven decisions from this
one flop solve: BB check, SB scenario bet, BB call; deal **2h**; BB check, SB scenario
bet, BB call; deal **3d**. The inspection runout is never supplied to flop training.
The panel shows action frequencies, direct sample coverage and both lines' pots
and remaining stacks. **View** opens each decision's full hand grid. **Export
comparison** saves all seven combo strategies, settings, starting ranges and the
reference observations. Comparing does not run training or local refinement.

The existing sizing rules produce the same 1.7 BB flop bet, then **6.3 BB on the
turn** instead of the screenshot's 16.3 BB. The resulting river pot is **21 BB**
with **89.5 BB** behind, versus 41 BB and 79.5 BB in the screenshot. The menus also
differ elsewhere. This is a reference for the same board/action types, not an
identical-game benchmark: sizes change incentives and reaching ranges.

The screenshot contains no action frequencies. Reference probabilities and
frequency errors are explicitly `null`; no accuracy score is invented. A numerical
GTO Wizard comparison needs its frequencies plus matching ranges, rake and action
tree. The app's preflop frequencies remain rounded to 10%. The report is also
available at `POST /api/jobs/<id>/comparison` with `{}` for this original flop root.

### Memory and device defaults

| Setting | Default | Purpose |
| --- | ---: | --- |
| `backend` | `tabular` | Select `deep_cfr` explicitly |
| `deep_device` | `cpu` | Or `mps` for Apple's GPU; unavailable MPS gives a clear error |
| `deep_hidden` | 128 | Two fully connected hidden layers |
| `deep_buffer_size` | 25,000 | Capacity of each of four uniform reservoirs |
| `deep_traversals` | 128 | Independent sampled deals per player per neural round |
| `deep_train_steps` | 256 | Minibatch optimizer steps per network fit |
| `deep_batch_size` | 256 | Training examples per optimizer step |
| `deep_inference_batch` | 32 | Concurrent traversals whose predictions are batched |

The four default replay arrays reserve **230,500,000 bytes (about 220 MiB)** total.
They store float32 features/targets/round weights and Boolean action masks. This
is not total process memory: PyTorch, optimizer states, card-evaluator cache,
visit tracking, retained solves and the OS also use memory. Replay size is capped
at 100,000 entries per buffer, network width at 256, and diagnostic state tracking
at 50,000 entries. GPU and CPU share the Mac's physical RAM. The app retains up to
two solves, so retaining two neural solves doubles their replay allocation.

CPU is a conservative default for these small networks, not a claim it is always
faster. Training temporarily uses at most two CPU threads. MPS batches both
traversal inference and network training; range inspection also batches predictions.
This does not move Python game simulation onto the GPU. Benchmark both devices
on the same scenario before choosing one. No cloud service or external bot is used.

### Algorithm and representation

- Independent external-sampling traversals enumerate the traverser's actions and
  sample the opponent's actions. Player updates alternate; a phase's policies
  remain fixed while all its samples are collected. Private deals are fresh for
  the next player's phase, avoiding the shared-deal update-order bias.
- Each player has an advantage reservoir and a separate strategy reservoir.
  Algorithm R gives every observed sample equal retention probability, including
  after capacity is reached. It is not a recent-samples sliding window.
- Stored advantage targets are sampled action value minus policy value, normalized
  by a **fixed root payoff scale**. Models are trained from scratch on accumulated
  reservoir data at each fit, using Adam, masked squared error, gradient clipping
  and linear CFR iteration weights `2 * sample_round / current_round`.
- Regret matching normalizes positive predicted advantages; if none are positive,
  it selects the largest predicted advantage (uniformly across ties). Round one
  starts uniformly. Illegal/padded actions are excluded from policies and losses.
- Strategy targets are recorded at sampled **opponent** nodes. Their visitation
  already accounts for the owner's reach; no extra reach factor is applied. Two
  separate softmax networks approximate the weighted average strategy. The final
  policy is not the latest advantage network. Average models publish together only
  after a complete round. An error retains the prior published policy but requires
  a new solve or refinement to train again.
- The 569 input features contain exact hole cards, unordered flop cards, separately
  ordered turn/river cards, rank/suit counts, public stacks/pot/commitments, position,
  street, public aggressor history, sizing profile, the full local action history,
  and the current action menu/sizes. Opponent hole cards and undealt cards never
  enter the encoder. Hand-strength bucketing is not used.
- Root ranges and prior local-refinement history are fixed context, not portable
  model inputs. Do not reuse a fitted model for different roots/ranges. Histories
  longer than 32 public tokens or menus larger than five actions fail explicitly
  rather than silently truncating information.

This implements the external-sampling, reservoir and average-policy structure of
[Deep CFR (Brown et al., ICML 2019)](https://proceedings.mlr.press/v97/brown19b.html)
with a much smaller model and training budget, separate average models per player,
and an incremental pause/resume scheduler. It is a research prototype, not a
reproduction of the paper's large-game performance. Function approximation can
introduce persistent error. A network can output plausible but poor strategies
in whole branches with little training data.

### Validate and compare

```sh
.venv/bin/python -m unittest discover -s tests -q
node --test tests/test_ui.mjs
PYTHONPATH=. .venv/bin/python benchmarks/deep_cfr.py --seconds 30 --seeds 7 23 --devices cpu mps --output deep-cfr-results.json
PYTHONPATH=. .venv/bin/python benchmarks/deep_cfr.py --street turn --seconds 30 --devices cpu --output deep-cfr-turn-results.json
PYTHONPATH=. .venv/bin/python benchmarks/deep_cfr.py --street flop --seconds 30 --devices cpu --output deep-cfr-flop-results.json
```

Run performance comparisons without other training jobs. The benchmark gives each
backend the same training-time budget and records version/device,
completed rounds/traversals, memory allocation, and direct versus predicted coverage.
On its small river fixture it computes exact best-response exploitability for the
**average policy**, before and after training. Flop/turn runs report no exploitability
estimate. A short neural run with no completed round is reported as unready; it
must not be treated as a trained strategy. Tests cover uniform replay retention,
legal-action masking, iteration weights, observation encoding, unseen-state
prediction labeling, exact resume equivalence, three-street flop traversal,
read-only reference inspection, API flop initialization/refinement/export, dependency-free
fallback, and measured improvement in the small river fixture. They do not establish
full Hold'em convergence or that Deep CFR beats tabular CFR at a fixed runtime.

### Recorded short-run validation

[Raw results](benchmarks/results/deep-cfr-river-m1-pro.json) record two seeds on an
M1 Pro with 16 GB RAM, using 15 seconds per run and the defaults above. Initial
exploitability was 0.96875 BB in this four-hand-pair river fixture.

| Backend | Seed 7 exploitability (BB) | Seed 23 exploitability (BB) | Completed neural rounds |
| --- | ---: | ---: | --- |
| Tabular | 0.0102 | 0.0102 | Not applicable |
| Deep CFR, CPU | 0.2265 | 0.2478 | 7 / 9 |
| Deep CFR, MPS | 0.4778 | 0.3428 | 2 / 3 |

CPU is the default based on this small-network comparison. Tabular CFR wins this
tiny game decisively; the test verifies that the neural implementation learns,
not that it is a replacement for tabular solving in small games. These brief
measurements do not evaluate unseen-branch accuracy or full-game playing strength.
