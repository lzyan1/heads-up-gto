"""Strict preflop state reconstruction and conditional chart frequencies."""
from dataclasses import dataclass
import math
import re
from .cards import COMBOS, RANKS, hand_class, parse_range, remove_blocked


@dataclass
class Preflop:
    history: tuple
    pot: float
    stacks: tuple  # BB/OOP, SB/IP
    contributions: tuple


def canonical_action(text):
    text = str(text).strip().lower()
    if text in ("check", "call", "fold"):
        return text
    match = re.fullmatch(r"(?:raise|bet|r)\s*[: ]\s*(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Invalid preflop action {text!r}. Use call, check, fold, or raise:2.5 (raise TO).")
    amount = float(match[1])
    if not math.isfinite(amount) or amount <= 0 or amount > 100:
        raise ValueError("Preflop raise-to amounts must be greater than 0 and at most 100 BB.")
    if abs(amount * 100 - round(amount * 100)) > 1e-6:
        raise ValueError("Use at most two decimal places for raise-to sizes.")
    return f"raise:{amount:g}"


def parse_history(value):
    if isinstance(value, str):
        value = re.split(r"[/,\n]+", value)
    if not isinstance(value, (list, tuple)) or not value or len(value) > 30:
        raise ValueError("Provide a complete preflop line, such as raise:2.5 / call.")
    return tuple(canonical_action(a) for a in value if str(a).strip())


def reconstruct(value):
    history = parse_history(value)
    paid = [1.0, .5]
    actor, last_raise, closed = 1, 1.0, False
    for index, action in enumerate(history):
        if closed:
            raise ValueError("Preflop actions continue after the betting round has ended.")
        other = 1 - actor
        facing = paid[other] - paid[actor]
        if action == "fold":
            raise ValueError("This hand ended preflop; there is no flop continuation.")
        if action == "check":
            if facing > 1e-8:
                raise ValueError("Cannot check while facing a bet.")
            if not (index == 1 and history[0] == "call"):
                raise ValueError("The only preflop check is BB checking after an SB limp.")
            closed = True
        elif action == "call":
            if facing <= 1e-8:
                raise ValueError("There is no outstanding bet to call.")
            paid[actor] = paid[other]
            closed = index > 0
        else:
            target = float(action.split(":")[1])
            increase = target - max(paid)
            if increase <= 0 or (increase + 1e-8 < last_raise and target != 100):
                raise ValueError(f"Illegal raise-to {target:g}; minimum is {max(paid) + last_raise:g} BB (or all-in).")
            paid[actor] = target
            last_raise = max(last_raise, increase)
        actor = other
    if not closed:
        raise ValueError("Finish the preflop line with a call, or limp / check.")
    if max(paid) >= 100:
        raise ValueError("Both players are all-in preflop; there are no postflop decisions.")
    return Preflop(history, sum(paid), tuple(100 - x for x in paid), tuple(paid))


def validate_chart(chart):
    if not isinstance(chart, dict) or chart.get("version") != 1:
        raise ValueError("Chart JSON must have version: 1.")
    required = {"stack_bb": 100, "small_blind_bb": .5, "big_blind_bb": 1, "ante_bb": 0, "rake": 0}
    if chart.get("game") != required:
        raise ValueError("Chart game must match 100 BB, 0.5/1 blinds, zero ante and zero rake. See the example schema.")
    nodes = chart.get("nodes")
    if not isinstance(nodes, dict) or not nodes or len(nodes) > 1000:
        raise ValueError("Chart must contain 1–1000 conditional action nodes.")
    compiled = {}
    for history, actions in nodes.items():
        key = "/".join(parse_history(history)) if history else ""
        if key != history or key in compiled:
            raise ValueError("Chart node keys must be canonical histories, e.g. raise:2.5/call.")
        if not isinstance(actions, dict) or not actions or len(actions) > 20:
            raise ValueError(f"Chart node {key!r} needs action-to-range mappings.")
        mapped = {}
        for action, weights in actions.items():
            canon = canonical_action(action)
            if action != canon:
                raise ValueError(f"Use canonical action {canon!r} in charts.")
            mapped[canon] = parse_range(weights)
        for hand in COMBOS:
            if sum(r.get(hand, 0) for r in mapped.values()) > 1 + 1e-8:
                raise ValueError(f"Action frequencies exceed 1 for {hand_class(hand)} at node {key!r}.")
        compiled[key] = mapped
    return compiled


def chart_ranges(chart, preflop, board):
    compiled = validate_chart(chart)
    ranges = [dict.fromkeys(COMBOS, 1.0), dict.fromkeys(COMBOS, 1.0)]
    history = []
    actor = 1
    for action in preflop.history:
        key = "/".join(history)
        if key not in compiled or action not in compiled[key]:
            raise ValueError(f"Chart has no frequencies for {action!r} after {key or 'the blinds'}. Add that node or use explicit ranges.")
        weights = compiled[key][action]
        ranges[actor] = {h: w * weights.get(h, 0) for h, w in ranges[actor].items() if w * weights.get(h, 0) > 0}
        history.append(action)
        actor = 1 - actor
    return tuple(remove_blocked(r, board) for r in ranges)


def demo_chart():
    """Synthetic fixture only. Never represented as an equilibrium chart."""
    classes = sorted(set(map(hand_class, COMBOS)))
    strength = {}
    for h in classes:
        a, b = RANKS.index(h[0]), RANKS.index(h[1])
        strength[h] = min(1.0, (a + b) / 24 + (.32 if a == b else .10 if h.endswith("s") else 0))
    def frequencies(fn):
        return {h: round(max(0, min(1, fn(s))), 3) for h, s in strength.items()}
    return {
        "version": 1, "name": "Illustrative demo — NOT GTO charts", "is_demo": True,
        "game": {"stack_bb": 100, "small_blind_bb": .5, "big_blind_bb": 1, "ante_bb": 0, "rake": 0},
        "nodes": {
            "": {"raise:2.5": frequencies(lambda s: .15 + .65 * s), "call": frequencies(lambda s: .18)},
            "call": {"check": frequencies(lambda s: .85 - .3 * s), "raise:4": frequencies(lambda s: .14 + .3 * s)},
            "call/raise:4": {"call": frequencies(lambda s: .15 + .7 * s)},
            "raise:2.5": {"call": frequencies(lambda s: .2 + .25 * s), "raise:9": frequencies(lambda s: max(0, s - .55))},
            "raise:2.5/raise:9": {"call": frequencies(lambda s: max(0, .8 * s - .2)), "raise:22": frequencies(lambda s: max(0, .7 * s - .5))},
            "raise:2.5/raise:9/raise:22": {"call": frequencies(lambda s: max(0, s - .4))},
        },
    }
