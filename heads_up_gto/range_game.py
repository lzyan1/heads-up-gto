"""Exact-card range sampling and read-only navigation for Deep CFR.

This context contains no training algorithm or stored strategy table. Policies
are supplied by the neural solver; both players' ranges are conditioned on
public actions and non-overlapping cards.
"""
from bisect import bisect_left
from dataclasses import asdict
import random
from .cards import DECK, card_text, combo_text, hand_class, remove_blocked
from .game import PLAYERS, STREETS
from .sizing import choose_sizes


class Distribution:
    def __init__(self, weights):
        self.hands = tuple(weights)
        self.weights = tuple(weights.values())
        total = 0.0
        self.cdf = []
        for w in self.weights:
            total += w
            self.cdf.append(total)
        self.total = total

    def draw(self, rng):
        return self.hands[bisect_left(self.cdf, rng.random() * self.total)]


class RangeGame:
    def __init__(self, root, ranges, settings):
        self.root, self.ranges, self.settings = root, ranges, settings
        if any(not r for r in ranges):
            raise ValueError("Both ranges need at least one hand after removing board cards.")
        self.distributions = [Distribution(r) for r in ranges]
        # Sample BB from its correct marginal after accounting for SB blockers.
        other_total = sum(ranges[1].values())
        card_mass = [0.0] * 52
        for (a, b), w in ranges[1].items():
            card_mass[a] += w
            card_mass[b] += w
        compatible = {}
        for (a, b), w in ranges[0].items():
            mass = other_total - card_mass[a] - card_mass[b] + ranges[1].get((a, b), 0)
            if mass > 1e-10:
                compatible[(a, b)] = w * mass
        if not compatible:
            raise ValueError("The two ranges have no legal, non-overlapping pair of hands.")
        self.first = Distribution(compatible)
        self.rng = random.Random(settings.seed)

    def _opponent_hand(self, player, blocked, rng):
        dist = self.distributions[1 - player]
        for _ in range(40):
            hand = dist.draw(rng)
            if not blocked.intersection(hand):
                return hand
        compatible = {h: w for h, w in self.ranges[1 - player].items() if not blocked.intersection(h)}
        return Distribution(compatible).draw(rng) if compatible else None

    def sample_deal(self):
        h0 = self.first.draw(self.rng)
        h1 = self._opponent_hand(0, set((*h0, *self.root.board)), self.rng)
        blocked = set((*h0, *h1, *self.root.board))
        rest = self.rng.sample([c for c in DECK if c not in blocked], 5 - len(self.root.board))
        return (h0, h1), (*self.root.board, *rest)

    def info_key(self, state, hand):
        return (state.actor, state.path, tuple(sorted(hand)), state.board)

    def policy(self, state, hand):
        raise NotImplementedError("A neural average policy is required.")

    def prepare_decision(self, state, ranges):
        pass

    def prepare_policies(self, state, hands):
        raise NotImplementedError("A neural average policy is required.")

    def locate(self, path):
        """Condition ranges along the stored policy; disclose untrained fallbacks."""
        state = self.root
        ranges = [dict(r) for r in self.ranges]
        uncertain = 0.0
        for step in path:
            if state.terminal:
                raise ValueError("The hand has already ended.")
            if state.pending:
                if not step.startswith("deal:"):
                    raise ValueError("Enter the next board card before taking an action.")
                from .cards import parse_cards
                card = parse_cards(step[5:], 1)[0]
                state = state.deal(card)
                ranges = [remove_blocked(r, state.board) for r in ranges]
                continue
            actions = {a.key: a for a in state.actions(self.settings)}
            if step not in actions:
                raise ValueError(f"Illegal action {step!r} at this node.")
            action_index = list(actions).index(step)
            actor = state.actor
            updated = {}
            self.prepare_decision(state, ranges)
            self.prepare_policies(state, ranges[actor])
            for hand, weight in ranges[actor].items():
                probs, _, _ = self.policy(state, hand)
                if probs is None:
                    # The initialized deterministic policy chooses the first
                    # legal action; never invent fractional propagation.
                    probs = [1.] + [0.] * (len(actions) - 1)
                    uncertain += weight
                if probs[action_index] > 0:
                    updated[hand] = weight * probs[action_index]
            ranges[actor] = updated
            state = state.apply(actions[step])
        return state, ranges, uncertain > 0

    def timeline(self, path):
        """Reconstruct public action panels without evaluating private strategies."""
        state, prefix, result = self.root, [], []
        for step in path:
            if state.pending:
                from .cards import parse_cards
                if not step.startswith("deal:"):
                    raise ValueError("Deal the next board card first.")
                state = state.deal(parse_cards(step[5:], 1)[0])
                result.append({"kind": "board", "street": STREETS[state.street], "board": [card_text(c) for c in state.board], "pot": state.pot, "path": list(prefix), "card": step[5:]})
            else:
                choices = state.actions(self.settings)
                action = next((a for a in choices if a.key == step), None)
                if action is None:
                    raise ValueError("Invalid action in the navigation path.")
                result.append({"kind": "decision", "phase": "postflop", "street": STREETS[state.street], "actor": state.actor,
                               "stack": state.stacks[state.actor], "pot": state.pot, "actions": [a.as_dict() for a in choices], "path": list(prefix), "selected": step})
                state = state.apply(action)
            prefix.append(step)
        if not state.pending and not state.terminal:
            result.append({"kind": "decision", "phase": "postflop", "street": STREETS[state.street], "actor": state.actor,
                           "stack": state.stacks[state.actor], "pot": state.pot, "actions": [a.as_dict() for a in state.actions(self.settings)], "path": prefix, "selected": None})
        return result

    def inspect(self, path, selected=None):
        """Inspect the existing average strategy without changing it."""
        state, ranges, uncertain = self.locate(path)
        actions = state.actions(self.settings)
        response = {"board": [card_text(c) for c in state.board], "street": STREETS[state.street], "actor": state.actor, "actor_label": PLAYERS[state.actor], "pot": state.pot, "stacks": state.stacks, "pending": state.pending, "terminal": state.terminal, "winner": state.winner, "path": path, "actions": [a.as_dict() for a in actions], "rows": [], "aggregate": [], "selected": None, "fallback_used": uncertain > 0, "coverage": 0.0, "timeline": self.timeline(path), "phase": "postflop"}
        if actions and state.sizing_profile:
            response["sizing"] = asdict(choose_sizes(state))
        if not actions:
            return response
        actor, other = state.actor, 1 - state.actor
        other_total = sum(ranges[other].values())
        card_mass = [0.0] * 52
        for (a, b), w in ranges[other].items():
            card_mass[a] += w
            card_mass[b] += w
        rows, aggregate = [], [0.0] * len(actions)
        total, covered = 0.0, 0.0
        self.prepare_decision(state, ranges)
        self.prepare_policies(state, ranges[actor])
        for hand, weight in ranges[actor].items():
            mass = max(0, other_total - card_mass[hand[0]] - card_mass[hand[1]] + ranges[other].get(hand, 0))
            joint = weight * mass
            if joint <= 1e-12:
                continue
            probs, updates, averages = self.policy(state, hand)
            total += joint
            row = {"hand": combo_text(hand), "class": hand_class(hand), "weight": weight, "mass": joint, "probabilities": probs, "updates": updates, "averages": averages}
            if probs is not None:
                covered += joint
                for i, p in enumerate(probs):
                    aggregate[i] += joint * p
            rows.append(row)
        if total:
            for row in rows:
                row["mass"] /= total
        from .diagnostics import sample_summary
        response["training"] = sample_summary(rows)
        if selected is not None:
            response["selected"] = next((dict(row) for row in rows if row["hand"] == combo_text(tuple(sorted(selected)))), None)
        response.update(rows=rows, aggregate=[x / covered for x in aggregate] if covered else [], coverage=covered / total if total else 0.0)
        if selected and response["selected"] is None:
            response["selected_error"] = "That hand is blocked by the board, outside this player's current range, or has no compatible opposing hand."
        return response
