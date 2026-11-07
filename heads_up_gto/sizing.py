"""Fixed public-state postflop sizing, shared by training and river solving.

Bets use the pot before betting. Raises are street-total raise-to multiples of
the opponent's bet. All-in replaces a prescribed size; it is never an extra
choice. Only non-donk river openings offer two sizes.
"""
from dataclasses import dataclass

MODEL_VERSION = "scenario-v3"
PROFILES = {
    "single_raised": (5.0, 97.5, 1),
    "three_bet": (23.0, 88.5, 0),
    "four_bet": (52.0, 74.0, 1),
}
BET_FRACTIONS = {
    "flop": {"cbet": (.33,), "donk": (.33,), "stab": (.33,)},
    "turn": {"barrel": (.66,), "probe": (.50,), "delayed_cbet": (.50,),
             "stab": (.50,), "donk": (.50,)},
    "river": {"barrel": (.33, 1.), "probe": (.33, 1.), "delayed_cbet": (.33, 1.),
              "stab": (.33, 1.), "donk": (.50,)},
}
RAISE_MULTIPLIERS = (3., 2.5, None)  # River raises are all-in.
SHOVE_REMAINING_POT_FRACTION = .50
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


@dataclass(frozen=True)
class SizingChoice:
    targets: tuple
    scenario: str
    reason: str


def _legal_target(state, target, cap, to_call):
    """Apply minimums, rounding, effective stacks, and the half-pot shove rule."""
    p, other = state.actor, 1 - state.actor
    maximum = state.paid[p] + cap
    minimum = state.paid[other] + state.last_raise if to_call else 1.
    # Preserve exact minimum raises and short all-ins despite 0.1 BB rounding.
    target = min(max(round(min(target, maximum), 1), minimum), maximum)
    amount = target - state.paid[p]
    # Use the pot if the opponent calls: no future opponent chips are added to
    # State.pot here. Compare the remaining *effective* stack, not a big stack's
    # uncallable excess. Equality does not trigger the strictly "less than" rule.
    called_pot = state.pot + amount + target - state.paid[other]
    remaining = round(cap - amount, 2)
    if remaining < SHOVE_REMAINING_POT_FRACTION * called_pot - 1e-9:
        target = maximum
    return round(target, 2)


def choose_sizes(state):
    """Return only the prescribed aggressive targets for this public state."""
    p, other = state.actor, 1 - state.actor
    to_call = round(max(0, state.paid[other] - state.paid[p]), 2)
    cap = round(min(state.stacks[p], state.stacks[other] + to_call), 2)
    if cap <= to_call + .001:
        return SizingChoice((), "facing_all_in", "Facing an all-in: fold or call.")
    if to_call:
        scenario = "reraise" if state.aggressions >= 2 else "check_raise" if p == 0 else "raise"
        if state.street == 0 and state.aggressions >= 2 and state.sizing_profile == "single_raised":
            return SizingChoice((), "reraise_pruned", "Single-raised pot: facing a flop raise, fold or call.")
        if state.aggressions >= 2 or state.street == 2:
            return SizingChoice((round(state.paid[p] + cap, 2),), scenario,
                                "River raises and permitted re-raises are all-in.")
        multiplier = RAISE_MULTIPLIERS[state.street]
        targets = (multiplier * state.paid[other],)
        reason = f"{SCENARIO_LABELS[scenario]} to {multiplier:g}× the opponent's bet."
    else:
        scenario = opening_scenario(state)
        fractions = BET_FRACTIONS[("flop", "turn", "river")[state.street]][scenario]
        targets = tuple(state.paid[p] + state.pot * fraction for fraction in fractions)
        sizes = " or ".join(f"{fraction:.0%}" for fraction in fractions)
        reason = f"{SCENARIO_LABELS[scenario]}: {sizes} of the pot before betting."
    legal = tuple(dict.fromkeys(_legal_target(state, t, cap, to_call) for t in targets))
    if round(state.paid[p] + cap, 2) in legal:
        reason += " Stack limits or less than half the called pot remaining: use all-in."
    return SizingChoice(legal, scenario, reason)


def sizing_metadata(profile):
    return {
        "model": MODEL_VERSION, "profile": profile,
        "source": "User-defined postflop sizing table; not optimized GTO sizes",
        "board_independent": True, "max_aggressive_actions_per_decision": 2,
        "max_aggressive_actions_by_street": {"flop": 1, "turn": 1, "river": 2},
        "all_in_available": "Only as the prescribed raise/re-raise or a replacement under stack/shove rules",
        "raise_sequence": "Flop: 3x, then shove except in single-raised pots (re-raise pruned). "
                          "Turn: 2.5x, then shove. River: shove.",
        "bet_fractions": BET_FRACTIONS,
        "raise_to_multipliers": {"flop": 3., "turn": 2.5, "river": "all_in"},
        "adjustments": "Legal minimums, effective stacks and 0.1 BB rounding; duplicate sizes merged",
        "jam_rules": {"remaining_called_pot_fraction_below": SHOVE_REMAINING_POT_FRACTION},
        "amounts": "BB; bets use the pot before betting; raises are total committed on the current street",
    }
