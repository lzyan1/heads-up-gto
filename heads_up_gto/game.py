"""Heads-up postflop betting tree. Amounts are BB, raise amounts are raise-to."""
from dataclasses import dataclass, replace
import math
from .sizing import choose_size

STREETS = ("Flop", "Turn", "River")
PLAYERS = ("BB · out of position", "SB · button")


@dataclass(frozen=True)
class Settings:
    sizes: tuple = ((.33, .75), (.5, 1.0), (.75, 1.25))
    raise_multiplier: float = 3.0
    max_raises: int = 1
    include_all_in: bool = True
    buckets: int = 8
    equity_samples: int = 24
    mode: str = "bucketed"
    chance_mode: str = "sampled"  # or enumerate_turns (exact cards only)
    seed: int = 7
    max_nodes: int | None = None  # unlimited unless an optional limit is chosen
    backend: str = "tabular"
    deep_device: str = "cpu"
    deep_hidden: int = 128
    deep_buffer_size: int = 25000  # per reservoir: two advantage + two strategy
    deep_traversals: int = 128  # per player, per neural CFR round
    deep_train_steps: int = 256  # per network fit
    deep_batch_size: int = 256
    deep_inference_batch: int = 32

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Settings must be an object.")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}.")
        data = dict(data)
        if "sizes" in data:
            data["sizes"] = tuple(tuple(sorted(set(float(x) for x in row))) for row in data["sizes"])
        obj = cls(**data)
        if obj.backend not in ("tabular", "deep_cfr"):
            raise ValueError("Backend must be tabular or deep_cfr.")
        if obj.deep_device not in ("cpu", "mps"):
            raise ValueError("Deep CFR device must be cpu or mps.")
        for name, lo, hi in (("deep_hidden", 32, 256), ("deep_buffer_size", 128, 100000),
                             ("deep_traversals", 1, 4096), ("deep_train_steps", 1, 4096),
                             ("deep_batch_size", 16, 2048), ("deep_inference_batch", 1, 128)):
            if type(getattr(obj, name)) is not int or not lo <= getattr(obj, name) <= hi:
                raise ValueError(f"{name} must be an integer between {lo} and {hi}.")
        if obj.backend == "deep_cfr":
            if obj.mode != "exact" or obj.chance_mode != "sampled":
                raise ValueError("Deep CFR uses exact card inputs and sampled boards.")
            if obj.max_nodes is not None:
                raise ValueError("Deep CFR uses bounded replay buffers; leave the decision limit blank.")
        if len(obj.sizes) != 3 or any(len(row) > 3 or any(not math.isfinite(x) or not .1 <= x <= 3 for x in row) for row in obj.sizes):
            raise ValueError("Give up to three pot fractions per street, each between 0.1 and 3.")
        if obj.chance_mode not in ("sampled", "enumerate_turns"):
            raise ValueError("Chance mode must be sampled or enumerate_turns.")
        if obj.chance_mode == "enumerate_turns" and obj.mode != "exact":
            raise ValueError("Turn enumeration requires exact cards.")
        if obj.mode not in ("bucketed", "exact"):
            raise ValueError("Mode must be bucketed or exact.")
        if type(obj.max_raises) is not int or not 0 <= obj.max_raises <= 2:
            raise ValueError("Raise cap must be 0, 1, or 2 raises after the opening bet.")
        if not math.isfinite(obj.raise_multiplier) or not 2 <= obj.raise_multiplier <= 4:
            raise ValueError("Raise multiplier must be between 2 and 4.")
        if obj.max_nodes is not None and (type(obj.max_nodes) is not int or obj.max_nodes < 1):
            raise ValueError("max_nodes must be null (unlimited) or a positive integer.")
        for name, lo, hi in (("buckets", 4, 32), ("equity_samples", 8, 256), ("seed", 0, 2**31 - 1)):
            value = getattr(obj, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer between {lo} and {hi}.")
        if type(obj.include_all_in) is not bool:
            raise ValueError("include_all_in must be true or false.")
        return obj


@dataclass(frozen=True)
class Action:
    key: str
    kind: str
    amount: float = 0  # chips paid now
    target: float = 0  # total paid this street
    label: str = ""

    def as_dict(self):
        return {"key": self.key, "kind": self.kind, "amount": self.amount, "target": self.target, "label": self.label}


@dataclass(frozen=True)
class State:
    base_pot: float
    start_stacks: tuple
    board: tuple
    street: int = 0
    actor: int = 0
    invested: tuple = (0.0, 0.0)
    paid: tuple = (0.0, 0.0)
    checked: bool = False
    aggressions: int = 0
    last_raise: float = 1.0
    path: tuple = ()
    pending: bool = False
    terminal: str = ""
    winner: int = -1
    sizing_profile: str = ""  # empty only for generic internal games/tests
    street_aggressor: int = -1
    aggressor_history: tuple = ()  # retained when local refinement resets path

    @property
    def pot(self):
        return round(self.base_pot + sum(self.invested), 2)

    @property
    def stacks(self):
        return tuple(round(s - i, 2) for s, i in zip(self.start_stacks, self.invested))

    def actions(self, settings):
        if self.pending or self.terminal:
            return ()
        p, other = self.actor, 1 - self.actor
        to_call = round(max(0, self.paid[other] - self.paid[p]), 2)
        cap = round(min(self.stacks[p], self.stacks[other] + to_call), 2)
        if to_call > 0:
            out = [Action("fold", "fold", label="Fold"), Action("call", "call", min(to_call, cap), label=f"Call {min(to_call, cap):g} BB")]
            minimum = self.paid[other] + self.last_raise
        else:
            out = [Action("check", "check", label="Check")]
            minimum = 1.0
        if self.sizing_profile:
            target = choose_size(self).target
            if target is None:
                return tuple(out)
            # Offer the scenario size and an immediate shove at every legal
            # decision. The shared target filter deduplicates jam-sized choices.
            targets = [target, self.paid[p] + cap]
        else:
            if cap <= to_call or to_call and (self.aggressions > settings.max_raises or self.stacks[other] <= 0):
                return tuple(out)
            targets = [max(minimum, self.paid[other] * settings.raise_multiplier)] if to_call else [max(1.0, self.pot * f) for f in settings.sizes[self.street]]
            if settings.include_all_in:
                targets.append(self.paid[p] + cap)
        seen = set()
        for target in targets:
            target = round(min(target, self.paid[p] + cap), 2)
            amount = round(target - self.paid[p], 2)
            if amount <= to_call or target in seen:
                continue
            seen.add(target)
            kind = "raise" if to_call else "bet"
            all_in = abs(amount - cap) < .001
            label = f"All-in {target:g} BB" if all_in else f"Raise to {target:g} BB" if to_call else f"Bet {target:g} BB · {amount / self.pot:.0%}"
            out.append(Action(f"{kind}:{target:g}", kind, amount, target, label))
        return tuple(out)

    def apply(self, action):
        p, other = self.actor, 1 - self.actor
        state = replace(self, path=(*self.path, action.key))
        if action.kind == "fold":
            return replace(state, terminal="fold", winner=other)
        invested, paid = list(self.invested), list(self.paid)
        invested[p] = round(invested[p] + action.amount, 2)
        paid[p] = round(paid[p] + action.amount, 2)
        state = replace(state, invested=tuple(invested), paid=tuple(paid), actor=other)
        if action.kind == "call" or action.kind == "check" and self.checked:
            if self.street == 2 or min(state.stacks) <= .001:
                return replace(state, terminal="showdown")
            return replace(state, pending=True)
        if action.kind == "check":
            return replace(state, checked=True)
        increase = action.target - max(self.paid)
        return replace(state, checked=False, aggressions=self.aggressions + 1, last_raise=max(self.last_raise, increase), street_aggressor=p)

    def deal(self, card):
        if not self.pending or card in self.board:
            raise ValueError("A new, unused board card is required at the end of a betting round.")
        return replace(self, street=self.street + 1, board=(*self.board, card), actor=0, paid=(0.0, 0.0), checked=False, aggressions=0, last_raise=1.0, pending=False, path=(*self.path, "/"), aggressor_history=(*self.aggressor_history, self.street_aggressor), street_aggressor=-1)

    def payoff(self, winner, player):
        if winner == -1:
            return (self.invested[1 - player] - self.invested[player]) / 2
        if winner == player:
            return self.base_pot / 2 + self.invested[1 - player]
        return -self.base_pot / 2 - self.invested[player]
