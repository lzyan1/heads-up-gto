# Heads up — postflop study

A local Python application for 100 BB heads-up no-limit hold'em: use the hardcoded GTO Wizard preflop charts, select a flop, and explore the remaining betting decisions. Displays a 13×13 range grid, exact-combo selection, mixed action frequencies, and per-decision training counts.

**This is an experimental approximate solver, not a certified GTO policy.** It implements the chart-conditioned postflop part of the proposed architecture. Preflop frequencies remain fixed. It does not implement a full-game preflop optimization loop or safe subgame solving.

## Run

Requires Python 3.11+. No runtime dependencies or internet connection are needed.

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

“Trained” coverage means the fraction of compatible range mass with average-strategy samples, not convergence. Combo tooltips include training counts. Missing chart frequencies and untrained postflop decisions have no invented recommendation. Action averages exclude unknown mass. Neither navigating the new interface nor adding training turns the prototype into a certified GTO solver.

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
