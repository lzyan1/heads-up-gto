"""Public-state sizing heuristics: one scenario size alongside an all-in.

These are deliberately small modeling choices, not solved optimal sizes. No
private hand, strategy frequency, or undealt card enters the choice. Tables
use pot fractions; the result is a single legal street-total amount in BB.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

MODEL_VERSION = "scenario-v2"
PROFILES = {
    "single_raised": (5.0, 97.5, 1),
    "three_bet": (23.0, 88.5, 0),
    "four_bet": (52.0, 74.0, 1),
}
# Columns: dry, wet, paired, three-suited. Donks intentionally use the larger
# abstraction requested by the user; large donks are not a universal GTO rule.
FLOP_FRACTIONS = {
    "single_raised": {"cbet": (.33, .67, .25, .25), "donk": (.75, 1., .50, .50)},
    "three_bet": {"cbet": (.33, .50, .25, .25), "stab": (.33, .50, .25, .25)},
    "four_bet": {"cbet": (.25, .33, .20, .20), "donk": (.50, .67, .33, .33)},
}
LATER_FRACTIONS = {
    "turn": {"barrel": .75, "probe": .75, "delayed_cbet": .67, "stab": .50, "donk": .33},
    "river": {"barrel": 1., "probe": .75, "delayed_cbet": .75, "stab": .67, "donk": .33},
}
SCENARIO_LABELS = {
    "cbet": "Continuation bet", "donk": "Donk lead", "stab": "Bet after a check",
    "barrel": "Barrel", "probe": "Probe after checks", "delayed_cbet": "Delayed continuation bet",
    "raise": "Raise", "check_raise": "Check-raise", "reraise": "Re-raise",
}


def profile_for(pot, stacks):
    for name, (start_pot, stack, _) in PROFILES.items():
        if abs(pot - start_pot) < .001 and len(stacks) == 2 and all(abs(s - stack) < .001 for s in stacks):
            return name
    raise ValueError("No sizing profile for this preflop pot and stacks.")


def _structure(board):
    ranks = {c // 4 + 2 for c in board}
    if 14 in ranks:
        ranks.add(1)  # Wheel connectivity, without treating K-A-2 as a straight.
    connected = max(sum(r in ranks for r in range(low, low + 5)) for low in range(1, 11))
    suited = max(Counter(c % 4 for c in board).values())
    paired = len({c // 4 for c in board}) < len(board)
    return connected, suited, paired


@lru_cache(maxsize=32768)
def board_features(board):
    """Coarse visible texture and whether the newest card changes the texture."""
    connected, suited, paired = _structure(board)
    if suited >= 4:
        texture = "four_flush"
    elif connected >= 4:
        texture = "four_straight"
    elif suited >= 3:
        texture = "flush"
    elif paired:
        texture = "paired"
    elif suited == 2 or connected >= 3:
        texture = "wet"
    else:
        texture = "dry"
    changed = False
    if len(board) > 3:
        old_connected, old_suited, _ = _structure(board[:-1])
        changed = (suited >= 3 and suited > old_suited or
                   connected >= 4 and connected > old_connected or
                   any(c // 4 == board[-1] // 4 for c in board[:-1]))
    return texture, bool(changed)


def opening_scenario(state):
    preflop_aggressor = PROFILES[state.sizing_profile][2]
    previous = state.aggressor_history[-1] if state.aggressor_history else -1
    if state.street == 0:
        if state.actor == preflop_aggressor:
            return "cbet"
        return "donk" if state.actor == 0 else "stab"
    if previous == state.actor:
        return "barrel"
    if previous != -1:
        return "donk" if state.actor == 0 else "stab"
    if state.actor == preflop_aggressor:
        return "delayed_cbet"
    return "probe" if state.actor == 0 else "stab"


def opening_fraction(state, scenario, texture, changed):
    if state.street == 0:
        index = ("dry", "wet", "paired", "flush").index(texture)
        return FLOP_FRACTIONS[state.sizing_profile][scenario][index]
    street = "turn" if state.street == 1 else "river"
    fraction = LATER_FRACTIONS[street][scenario]
    if texture == "wet" and scenario in ("probe", "delayed_cbet", "stab"):
        fraction = {"probe": 1., "delayed_cbet": .75, "stab": .67}[scenario]
    if state.street == 2 and scenario == "barrel":
        fraction = .75 if texture == "wet" else 1.
        if texture == "dry" and state.aggressor_history == (state.actor, state.actor):
            fraction = 1.25
    if state.street == 1 and state.sizing_profile == "four_bet":
        fraction = min(fraction, .50)
    if texture in ("paired", "flush") or changed:
        fraction = min(fraction, .50)
    if texture in ("four_flush", "four_straight"):
        fraction = min(fraction, .33)
    return fraction


@dataclass(frozen=True)
class SizingChoice:
    target: float | None
    scenario: str
    texture: str
    reason: str


def choose_size(state):
    """Choose the scenario size; State.actions also offers a legal all-in."""
    p, other = state.actor, 1 - state.actor
    to_call = round(max(0, state.paid[other] - state.paid[p]), 2)
    cap = round(min(state.stacks[p], state.stacks[other] + to_call), 2)
    texture, changed = board_features(state.board)
    if cap <= to_call + .001:
        return SizingChoice(None, "facing_all_in", texture, "Facing an all-in: fold or call.")
    if to_call:
        scenario = "reraise" if state.aggressions >= 2 else "check_raise" if p == 0 else "raise"
        minimum = state.paid[other] + state.last_raise
        if state.aggressions >= 2:
            target = state.paid[p] + cap
            reason = "A re-raise uses all-in to keep the tree compact."
        else:
            # Raise increment as a fraction of the pot AFTER matching the bet.
            fraction = ((.75, .50), (.75, .67), (1., .75))[state.street][p]
            if state.street == 0 and state.sizing_profile == "four_bet":
                fraction = .50
            target = max(minimum, state.paid[other] + fraction * (state.pot + to_call))
            reason = f"{SCENARIO_LABELS[scenario]}: add {fraction:.0%} of the pot after calling."
    else:
        scenario = opening_scenario(state)
        fraction = opening_fraction(state, scenario, texture, changed)
        target = max(1., state.pot * fraction)
        reason = f"{SCENARIO_LABELS[scenario]} · {texture.replace('_', ' ')} board · {fraction:.0%} pot."
        if cap / state.pot <= .75:
            target = state.paid[p] + cap
            reason += " Low stack-to-pot ratio: use all-in."
    target = round(min(target, state.paid[p] + cap), 1)
    # Round normal amounts to 0.1 BB; preserve a short all-in and full-raise
    # minimum exactly. No rounding may put more chips in than are available.
    minimum = state.paid[other] + state.last_raise if to_call else 1.
    target = min(max(target, minimum), state.paid[p] + cap)
    amount = target - state.paid[p]
    called_pot = state.pot + amount + target - state.paid[other]
    if amount >= .75 * cap or cap - amount <= .20 * called_pot:
        target = state.paid[p] + cap
        if "all-in" not in reason:
            reason += " Use all-in instead of leaving a small remainder."
    return SizingChoice(round(target, 2), scenario, texture, reason)


def sizing_metadata(profile):
    return {
        "model": MODEL_VERSION, "profile": profile, "source": "Heuristic scenario sizing; not GTO Wizard postflop data",
        "board_independent": False, "max_aggressive_actions_per_decision": 2,
        "all_in_available": "At every decision where aggression is legal; duplicate sizes are merged",
        "raise_sequence": "bet, one normal raise, then all-in re-raise",
        "flop_columns": ["dry", "wet", "paired", "three_suited"],
        "flop_fractions": FLOP_FRACTIONS[profile], "later_fractions": LATER_FRACTIONS,
        "adjustments": "Board texture, prior-street aggressor, position and effective stack; 0.1 BB rounding",
        "jam_rules": {"opening_spr_at_most": .75, "commitment_fraction_at_least": .75, "remaining_called_pot_fraction_at_most": .20},
        "amounts": "BB; raises are total committed on the current street",
    }
