"""External-sampling MCCFR with exact cards or a perfect-recall observation abstraction.

Deals use the correct joint product of range weights, conditioned on disjoint cards.
Future cards are sampled from the remaining deck. Both players' private observations
and public observations are retained across streets; no future card enters a key.
"""
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import random
import threading
import time
from .cards import DECK, card_text, combo_text, hand_class, remove_blocked, score
from .game import PLAYERS, STREETS
from .sizing import board_features, choose_size


class BudgetReached(Exception):
    pass


class Cancelled(Exception):
    pass


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


class Node:
    __slots__ = ("regrets", "total", "updates", "averages")

    def __init__(self, count):
        self.regrets = [0.0] * count
        self.total = [0.0] * count
        self.updates = 0
        self.averages = 0

    def strategy(self):
        positive = [max(r, 0.0) for r in self.regrets]
        total = sum(positive)
        return [r / total for r in positive] if total > 0 else [1 / len(positive)] * len(positive)

    def average(self):
        total = sum(self.total)
        return [v / total for v in self.total] if total > 0 else None


def board_observation(board):
    counts = sorted(Counter(c // 4 for c in board).values(), reverse=True)
    suits = max(Counter(c % 4 for c in board).values())
    # Preserve every public feature used by the sizing model. Otherwise boards
    # sharing a bucket could incorrectly share regrets for different bet sizes.
    return (tuple(counts), min(suits, 4), max(c // 4 for c in board) > max(c // 4 for c in board[:3]), board_features(board))


class Solver:
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
        self.nodes = {}
        self.observations = {}
        self.iterations = 0
        self.rng = random.Random(settings.seed)
        self.elapsed = 0.0
        self.run_elapsed = 0.0
        self.time_limit_seconds = None
        self.status = "ready"
        self.cancel = threading.Event()
        self._visits = 0
        self.attempted_iterations = 0
        self.completed_traversals = [0, 0]
        self.interrupted_iterations = 0

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

    def bucket(self, player, hand, board):
        cache_key = (player, hand, board)
        cached = self.observations.get(cache_key)
        if cached is not None:
            return cached
        seed = int.from_bytes(hashlib.blake2b(repr((self.settings.seed, cache_key)).encode(), digest_size=8).digest(), "big")
        rng = random.Random(seed)
        blocked = set((*hand, *board))
        wins, count = 0.0, 0
        for _ in range(self.settings.equity_samples):
            other = self._opponent_hand(player, blocked, rng)
            if other is None:
                break
            used = blocked.union(other)
            runout = (*board, *rng.sample([c for c in DECK if c not in used], 5 - len(board)))
            a, b = score(hand, runout), score(other, runout)
            wins += 1 if a > b else .5 if a == b else 0
            count += 1
        result = min(self.settings.buckets - 1, int(wins / max(count, 1) * self.settings.buckets))
        self.observations[cache_key] = result
        return result

    def info_key(self, state, hand, player=None):
        player = state.actor if player is None else player
        if self.settings.mode == "exact":
            return (player, state.path, hand, state.board)
        # Retain previous private buckets and public textures (perfect recall in the
        # abstract observation game). Hidden opponent cards never enter the key.
        lengths = range(len(self.root.board), len(state.board) + 1)
        private = tuple(self.bucket(player, hand, state.board[:n]) for n in lengths)
        public = tuple(board_observation(state.board[:n]) for n in lengths)
        return (player, state.path, private, public)

    def _walk(self, state, hands, runout, winner, traverser):
        self._visits += 1
        if self._visits % 256 == 0 and self.cancel.is_set():
            raise Cancelled()
        if state.terminal:
            return state.payoff(state.winner if state.terminal == "fold" else winner, traverser)
        if state.pending:
            return self._walk(state.deal(runout[len(state.board)]), hands, runout, winner, traverser)
        actions = state.actions(self.settings)
        key = self.info_key(state, hands[state.actor])
        node = self.nodes.get(key)
        if node is None:
            if self.settings.max_nodes is not None and len(self.nodes) >= self.settings.max_nodes:
                raise BudgetReached()
            node = self.nodes[key] = Node(len(actions))
        strategy = node.strategy()
        if state.actor == traverser:
            values = [self._walk(state.apply(a), hands, runout, winner, traverser) for a in actions]
            value = sum(p * v for p, v in zip(strategy, values))
            for i, v in enumerate(values):
                node.regrets[i] += v - value
            node.updates += 1
            return value
        # Opponent actions are sampled from current strategy. Their own reach is
        # represented by visitation; the traverser's actions are all enumerated.
        for i, p in enumerate(strategy):
            node.total[i] += p
        node.averages += 1
        action = self.rng.choices(actions, weights=strategy, k=1)[0]
        return self._walk(state.apply(action), hands, runout, winner, traverser)

    def train(self, iterations=None, progress=None, *, time_limit_seconds=None):
        if (iterations is None) == (time_limit_seconds is None):
            raise ValueError("Choose either an iteration count or a time limit.")
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError("Iterations must be a positive integer.")
        if time_limit_seconds is not None and (type(time_limit_seconds) not in (int, float) or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError("Time limit must be a positive, finite number of seconds.")
        start = time.monotonic()
        previous = self.elapsed
        self.run_elapsed = 0.0
        self.time_limit_seconds = time_limit_seconds
        self.status = "running"
        target = self.iterations + iterations if iterations is not None else None
        deadline = start + time_limit_seconds if time_limit_seconds is not None else None
        last_report = start
        in_iteration = False
        try:
            while target is None or self.iterations < target:
                if self.cancel.is_set():
                    raise Cancelled()
                # Complete both players' traversals before stopping. This avoids
                # repeatedly truncating the same expensive part of an iteration.
                if deadline is not None and time.monotonic() >= deadline:
                    break
                self.attempted_iterations += 1
                in_iteration = True
                hands, runout = self.sample_deal()
                a, b = score(hands[0], runout), score(hands[1], runout)
                winner = 0 if a > b else 1 if b > a else -1
                for player in (0, 1):
                    self._walk(self.root, hands, runout, winner, player)
                    self.completed_traversals[player] += 1
                self.iterations += 1
                in_iteration = False
                now = time.monotonic()
                if progress and now - last_report > .5:
                    self.run_elapsed = now - start
                    self.elapsed = previous + self.run_elapsed
                    progress(self.stats(target))
                    last_report = now
            self.status = "complete"
        except BudgetReached:
            self.status = "node_limit"
        except Cancelled:
            self.status = "cancelled"
        except MemoryError:
            # Real RAM exhaustion is possible even without an application cap.
            # Keep accumulated strategy available instead of failing the job.
            self.status = "memory_exhausted"
        finally:
            if in_iteration:
                self.interrupted_iterations += 1
            self.run_elapsed = time.monotonic() - start
            self.elapsed = previous + self.run_elapsed
        return self.stats(target)

    def stats(self, target=None):
        return {"status": self.status, "iterations": self.iterations, "target": target, "nodes": len(self.nodes), "seconds": round(self.elapsed, 2), "run_seconds": round(self.run_elapsed, 2), "time_limit_seconds": self.time_limit_seconds, "mode": self.settings.mode, "exploitability": None, "attempted_iterations": self.attempted_iterations, "completed_traversals": list(self.completed_traversals), "interrupted_iterations": self.interrupted_iterations, "partial_updates_possible": self.interrupted_iterations > 0, "tree_visits": self._visits, "max_nodes": self.settings.max_nodes}

    def coverage_diagnostics(self):
        """On-demand scan of stored nodes; unseen information sets are not counted."""
        from .diagnostics import sample_summary
        def street_index(key):
            return len(key[3]) - 3 if self.settings.mode == "exact" else self.root.street + len(key[3]) - 1
        # Stream rows into histograms instead of duplicating millions of nodes.
        streets = {name: sample_summary(
            {"updates": node.updates, "averages": node.averages}
            for key, node in self.nodes.items() if street_index(key) == index)
            for index, name in enumerate(STREETS)}
        return {"scope": "stored information sets only; excludes unseen nodes", "streets": streets}

    def policy(self, state, hand):
        node = self.nodes.get(self.info_key(state, hand))
        avg = node.average() if node and node.updates > 0 else None
        return avg, node.updates if node else 0, node.averages if node else 0

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
            for hand, weight in ranges[actor].items():
                probs, _, _ = self.policy(state, hand)
                if probs is None:
                    # Uniform is the actual initialized strategy at an unvisited
                    # information set; disclose its use in range propagation.
                    probs = [1 / len(actions)] * len(actions)
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
            response["sizing"] = asdict(choose_size(state))
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
