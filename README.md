# Heads up — Deep CFR study

A local Python application for 100 BB heads-up no-limit hold’em. Follow the fixed
GTO Wizard preflop charts, select a flop, and study the learned strategy across
flop, turn and river. The browser shows a 13×13 range grid, individual suit
combinations, mixed action frequencies, and direct sample counts.

**Deep CFR is the only training method.** It learns from exact cards and betting
history using neural advantage and average-policy networks. This is an
experimental approximate solver, not a certified GTO policy. Preflop charts
remain fixed; there is no full-game preflop optimization or safe subgame solving.

## Install and run

Requires Python 3.11+, NumPy and PyTorch. Both libraries are required dependencies
and are installed with the project:

```sh
python3 -m venv .venv                 # only if this checkout has no environment
.venv/bin/python -m pip install -e .
.venv/bin/python run.py --open
```

Or double-click `run.command` on macOS after installing. The interface opens at
http://127.0.0.1:8000. Stop with Control-C; use `--port 8001` for another port.
The server binds only to loopback, and training works offline after installation.

## Workflow

1. Follow SB/BB preflop actions in the horizontal history strip. The six supplied
   GTO Wizard charts, rounded to 10%, are the only preflop source.
2. After a closing call, choose three flop cards and **Solve continuation**.
   A flop-root solve trains all three streets together with sampled future cards.
3. Follow postflop actions and choose turn/river cards to inspect the same learned
   average strategy. Revisiting a decision preserves later actions; choosing a
   different action or card replaces the later branch.
4. Hover over the range grid to inspect exact suit combinations. Click to pin a
   hand; arrow keys also navigate. **Strategy** shows conditional action mixes,
   **Ranges** shows reach by bar height, and **Breakdown** combines both.
5. **Settings** offers a time limit (60 seconds by default) or a number of neural
   rounds, plus CPU/Apple GPU, network width, replay capacity, training steps,
   batch sizes and seed. Settings apply to new solves and refinements.
6. **Train more** preserves the existing networks, replay buffers, random states
   and optimizer work. Stop/time budgets pause between traversal batches or
   optimizer steps, including within a round. Predictions first appear after
   a complete round; later pauses preserve the last completed average strategy.
7. **Refine decision** starts a separate local solve using conditioned ranges and
   the current settings. Before a complete policy exists, uniform propagation is
   disclosed. Refinement is not safe subgame solving.
8. **Export** saves displayed strategies, configuration, root state/ranges and
   diagnostics as JSON. It does not save a resumable neural checkpoint. Training
   lives in memory and restarting loses it. Only the two most recent solves are
   retained; expired older decisions in a refined history become unavailable.

Green is check/call, blue is fold, reds are bets/raises, and dark red is all-in.
Recommendations apply to the current acting player. Raises are total street
amounts in BB. Cards use `s` ♠, `h` ♥, `d` ♦ and `c` ♣.

Neural predictions include unseen hands. **Direct samples** measures compatible
range mass with both advantage and average-policy visits; it is separate from
prediction availability and does not measure accuracy or convergence. Once the
50,000-state visit tracker fills, counts/coverage are lower bounds. Replay
sampling and neural learning continue normally.

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

## Training and API

The API accepts either `time_limit_seconds` (positive finite seconds) or
`iterations` (a positive integer counting complete neural rounds), never both.
Omitting both selects 60 seconds. Each solve/refinement/resume gets a fresh
budget; `seconds` and completed rounds are cumulative, while `run_seconds`
measures the current run. A time budget can overrun by the current batch or
optimizer step, plus device initialization. Preparation and range inspection
are outside the training timer.

```json
{
  "flop": "As Kd 7c",
  "preflop": "raise:2.5/call",
  "time_limit_seconds": 60,
  "settings": {"deep_device": "cpu", "seed": 7}
}
```

Send this to `POST /api/solve`; inspect progress at `GET /api/jobs/<id>`.
Use `POST /api/jobs/<id>/node` with `{"path": []}` to inspect the root after
training stops. Paths contain action keys and `deal:2h` board selections.
`resume`, `stop`, `refine`, `export`, `comparison`, and `diagnostics` are also
job endpoints. Settings use only neural and internal betting-tree fields;
removed backend, bucketing, turn-enumeration and decision-limit settings are
rejected rather than silently changing the solve. `resume` preserves settings.

`POST /api/jobs/<id>/diagnostics` summarizes bounded visit counters by street;
node inspection includes unseen compatible hands and sample distributions.
The descriptive 200-sample threshold is a visibility aid, not a confidence bound.
No full-game exploitability is computed. `diagnostics.river_exploitability`
evaluates exact best responses for small river fixtures (at most 400 legal hand
pairs). Zero-sum terminal utilities use exact sampled cards and hand evaluation.

A failed neural round retains the previously published average strategy but
requires a new solve or refinement to train again. Buffers are bounded, but
networks, evaluator caches, retained solves and process overhead also use RAM.
Local refinement does not preserve opponent counterfactual-value guarantees.
Preflop transcription, action restrictions, sampling and function approximation
all limit the result. The app does not connect to a poker client.

## Deep CFR

### Supplied GTO Wizard line

In **Settings → Load a specific spot**, choose **Set up supplied GTO Wizard flop**.
This fills in SB raise 2.5 / BB call and **As Kd 7c** without starting
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
device the same training-time budget and records version/device,
completed rounds/traversals, memory allocation, and direct versus predicted coverage.
On its small river fixture it computes exact best-response exploitability for the
**average policy**, before and after training. Flop/turn runs report no exploitability
estimate. A short neural run with no completed round is reported as unready; it
must not be treated as a trained strategy. Tests cover uniform replay retention,
legal-action masking, iteration weights, observation encoding, unseen-state
prediction labeling, exact resume equivalence, three-street flop traversal,
read-only reference inspection, API flop initialization/refinement/export, and
measured improvement in the small river fixture. They do not establish full
Hold'em convergence or unseen-branch accuracy.

## Code map

- `heads_up_gto/deep_cfr.py`: the sole training implementation, neural models,
  replay reservoirs, pause/resume and prediction diagnostics.
- `heads_up_gto/range_game.py`: compatible private-deal sampling, exact-state
  keys, range conditioning and read-only navigation. Contains no trainer.
- `heads_up_gto/game.py` / `sizing.py`: betting legality and scenario sizes.
- `heads_up_gto/cards.py`: cards, ranges and exact hand evaluation.
- `heads_up_gto/preflop.py` / `screenshot_charts.py`: fixed chart boundary.
- `heads_up_gto/server.py`: local HTTP API and one background training job.
- `heads_up_gto/diagnostics.py` / `reference_line.py`: sample summaries, small
  river best responses and read-only screenshot-line inspection.
- `heads_up_gto/static/`: browser interface with no remote assets or scripts.

The original experiment’s [recorded measurements](benchmarks/results/deep-cfr-river-m1-pro.json)
are retained as historical data. The current benchmark runs Deep CFR only.
