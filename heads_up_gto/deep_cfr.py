"""Deterministic neural flop/turn training with direct mixed river boundaries.

Earlier-street sampled advantages train bounded replay models. All executed
flop/turn policies select one action. River policies and continuation values
come from range-conditioned sequence-form LPs, never neural regret matching.
This restricted hybrid algorithm has no full-game equilibrium certificate.
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
import random
import time
import threading

import numpy as np
import torch
from torch import nn

from .cards import score, remove_blocked
from .game import Settings
from .range_game import RangeGame
from .river_jobs import RiverJob, RiverExecutor
from .river_only import exact_view

MAX_ACTIONS = 5
MAX_HISTORY = 32
CARD_WIDTH = 52 + 13 + 4
FEATURES = 4 * CARD_WIDTH + 34 + MAX_HISTORY * 7 + MAX_ACTIONS * 7
TRACKING_LIMIT = 50000
KINDS = ("fold", "check", "call", "bet", "raise")


def encode(state, hand, actions, scale):
    """Exact public history and private observation, with explicit action sizes.

    Root ranges are fixed for the lifetime of each solver. Models must never be
    reused for a new root/range distribution. Flop/hole order is immaterial;
    turn and river order, and the entire local betting history, are retained.
    """
    if len(state.path) > MAX_HISTORY or len(actions) > MAX_ACTIONS:
        raise ValueError("State exceeds the neural encoder's history/action capacity.")
    result = np.zeros(FEATURES, dtype=np.float32)
    for group, cards in enumerate((hand, state.board[:3], state.board[3:4], state.board[4:5])):
        offset = group * CARD_WIDTH
        for card in cards:
            result[offset + card] = 1
            result[offset + 52 + card // 4] += 1 / 3
            result[offset + 65 + card % 4] += 1 / 3
    numeric = [float(state.actor == p) for p in (0, 1)]
    numeric += [float(state.street == s) for s in range(3)]
    numeric += [state.pot / scale, *(v / scale for v in state.stacks),
                *(v / scale for v in state.paid), *(v / scale for v in state.invested),
                state.last_raise / scale, float(state.checked), state.aggressions / 4]
    numeric += [float(state.street_aggressor == p) for p in (-1, 0, 1)]
    for street in range(3):
        prior = state.aggressor_history[street] if street < len(state.aggressor_history) else -2
        numeric += [float(prior == p) for p in (-2, -1, 0, 1)]
    numeric += [float(state.sizing_profile == p) for p in ("", "single_raised", "three_bet", "four_bet")]
    offset = 4 * CARD_WIDTH
    result[offset:offset + 34] = numeric
    offset += 34
    for index, token in enumerate(state.path):
        kind, _, amount = token.partition(":")
        result[offset + index * 7 + (*KINDS, "/").index(kind)] = 1
        if amount:
            result[offset + index * 7 + 6] = float(amount) / scale
    offset += MAX_HISTORY * 7
    for index, action in enumerate(actions):
        result[offset + index * 7 + KINDS.index(action.kind)] = 1
        result[offset + index * 7 + 5] = action.amount / scale
        result[offset + index * 7 + 6] = action.target / scale
    return result


class Reservoir:
    """Algorithm R; compact CPU arrays, never a sliding window of recent play."""
    def __init__(self, capacity, seed):
        self.capacity, self.seen, self.size = capacity, 0, 0
        self.rng = random.Random(seed)
        self.features = np.empty((capacity, FEATURES), dtype=np.float32)
        self.targets = np.empty((capacity, MAX_ACTIONS), dtype=np.float32)
        self.masks = np.empty((capacity, MAX_ACTIONS), dtype=np.bool_)
        self.rounds = np.empty(capacity, dtype=np.float32)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.features, self.targets, self.masks, self.rounds))

    def add(self, features, targets, iteration):
        self.seen += 1
        slot = self.seen - 1 if self.seen <= self.capacity else self.rng.randrange(self.seen)
        if slot >= self.capacity:
            return
        n = len(targets)
        self.features[slot] = features
        self.targets[slot] = 0
        self.targets[slot, :n] = targets
        self.masks[slot] = False
        self.masks[slot, :n] = True
        self.rounds[slot] = iteration
        self.size = min(self.capacity, self.size + 1)


class Network(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURES, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, MAX_ACTIONS))

    def forward(self, features):
        return self.layers(features)


def greedy_policy(values):
    """One deterministic action; stable ties choose the first legal action."""
    result = np.zeros(len(values), dtype=np.float32)
    result[int(np.argmax(values))] = 1.
    return result


def weighted_loss(predictions, targets, masks, rounds, iteration, strategy=False):
    if strategy:
        predictions = torch.softmax(predictions.masked_fill(~masks, -1e9), dim=1)
    errors = ((predictions - targets).square() * masks).sum(dim=1)
    return (errors * (2 * rounds / iteration)).mean()


@dataclass(slots=True)
class SampleCounts:
    updates: int = 0
    averages: int = 0


@dataclass
class RiverRequest:
    state: object
    player: int
    hand: tuple


class DeepCFRSolver(RangeGame):
    def __init__(self, root, ranges, settings):
        Settings.from_dict(vars(settings))
        if (root.street not in (0, 1) or len(root.board) != root.street + 3
                or root.pending or root.terminal):
            raise ValueError("Neural training starts at an active flop or turn; rivers use direct solving.")
        super().__init__(root, ranges, settings)
        if settings.deep_device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this Python environment. Choose CPU or install an MPS-enabled PyTorch build.")
        self.nodes = {}  # bounded visit counters, never a strategy table
        self.iterations = 0
        self.attempted_iterations = 0
        self.completed_traversals = [0, 0]
        self._visits = 0
        self.elapsed = 0.0
        self.run_elapsed = 0.0
        self.time_limit_seconds = None
        self.status = "ready"
        self.cancel = threading.Event()
        self.device = torch.device(settings.deep_device)
        self.scale = max(1.0, root.base_pot / 2 + max(root.start_stacks))
        self.advantage_memory = [Reservoir(settings.deep_buffer_size, settings.seed + 100 + p) for p in (0, 1)]
        self.strategy_memory = [Reservoir(settings.deep_buffer_size, settings.seed + 200 + p) for p in (0, 1)]
        self.advantages = [None, None]
        self.average_models = [None, None]
        self.batch_rng = np.random.default_rng(settings.seed)
        self.phase = "ready"
        self.phase_step = 0
        self.phase_total = 0
        self.losses = {}
        self.tracking_saturated = False
        self._work = None
        self._failed = False
        self._policy_cache = {}
        self._river_cache = OrderedDict()
        self._inspection_rivers = OrderedDict()
        self._river_executor = RiverExecutor()
        self._river_job = None
        self._river_job_key = None
        self.river_solves = self.river_cache_hits = self.river_off_path_solves = 0
        self.river_seconds = 0.
        self.last_river_certificate = None
        self.river_active = False

    def _encode(self, state, hand, actions):
        return encode(state, hand, actions, self.scale)

    def _record(self, state, hand, average):
        key = self.info_key(state, hand)
        counts = self.nodes.get(key)
        if counts is None:
            if len(self.nodes) >= TRACKING_LIMIT:
                self.tracking_saturated = True
                return
            counts = self.nodes[key] = SampleCounts()
        if average:
            counts.averages += 1
        else:
            counts.updates += 1

    def _deal(self):
        hands, runout = self.sample_deal()
        a, b = score(hands[0], runout), score(hands[1], runout)
        return hands, runout, 0 if a > b else 1 if b > a else -1

    def _traverse(self, state, hands, runout, winner, traverser, iteration):
        self._visits += 1
        if state.terminal:
            return state.payoff(state.winner if state.terminal == "fold" else winner, traverser) / self.scale
        if state.pending:
            return (yield from self._traverse(state.deal(runout[len(state.board)]), hands, runout,
                                              winner, traverser, iteration))
        if state.street == 2:
            value = yield RiverRequest(state, traverser, hands[traverser])
            return value / self.scale
        actions = state.actions(self.settings)
        features = self._encode(state, hands[state.actor], actions)
        probabilities = yield (state.actor, features, len(actions))
        if state.actor == traverser:
            values = []
            for action in actions:
                values.append((yield from self._traverse(state.apply(action), hands, runout, winner,
                                                        traverser, iteration)))
            values = np.asarray(values, dtype=np.float32)
            value = float(np.dot(probabilities, values))
            self.advantage_memory[traverser].add(features, values - value, iteration)
            self._record(state, hands[state.actor], False)
            return value
        # Opponent reach is represented by external-sampling visitation. Adding
        # an extra reach factor here would bias the average-policy targets.
        self.strategy_memory[state.actor].add(features, probabilities, iteration)
        self._record(state, hands[state.actor], True)
        index = self.rng.choices(range(len(actions)), weights=probabilities, k=1)[0]
        return (yield from self._traverse(state.apply(actions[index]), hands, runout, winner,
                                          traverser, iteration))

    def _collect(self, player, iteration):
        self._river_cache.clear()  # policies change only between player phases
        remaining = self.settings.deep_traversals
        active = []
        while remaining or active:
            while remaining and len(active) < self.settings.deep_inference_batch:
                hands, runout, winner = self._deal()
                traversal = self._traverse(self.root, hands, runout, winner, player, iteration)
                active.append((traversal, next(traversal)))
                remaining -= 1
            predictions = [None] * len(active)
            for actor in (0, 1):
                indices = [i for i, (_, request) in enumerate(active) if not isinstance(request, RiverRequest) and request[0] == actor]
                if not indices:
                    continue
                model = self.advantages[actor]
                if model is None:
                    values = np.zeros((len(indices), MAX_ACTIONS), dtype=np.float32)
                else:
                    features = np.stack([active[i][1][1] for i in indices])
                    with torch.inference_mode():
                        values = model(torch.from_numpy(features).to(self.device)).cpu().numpy()
                for row, index in enumerate(indices):
                    predictions[index] = greedy_policy(values[row, :active[index][1][2]])
            pending = []
            for (traversal, request), probabilities in zip(active, predictions):
                if isinstance(request, RiverRequest):
                    probabilities = yield from self._river_value(request)
                try:
                    pending.append((traversal, traversal.send(probabilities)))
                except StopIteration:
                    self.completed_traversals[player] += 1
                    self.phase_step += 1
            active = pending
            yield

    def _training_policy(self, state, hands):
        actions = state.actions(self.settings)
        model = self.advantages[state.actor]
        hands = list(hands)
        result = []
        for start in range(0, len(hands), 256):
            batch = hands[start:start + 256]
            if model is None:
                result.extend([0] * len(batch))
            else:
                features = np.stack([self._encode(state, h, actions) for h in batch])
                with torch.inference_mode():
                    values = model(torch.from_numpy(features).to(self.device))[:, :len(actions)]
                    result.extend(values.argmax(dim=1).cpu().tolist())
        return dict(zip(hands, result))

    def _river_ranges(self, target):
        state = self.root
        ranges = [dict(r) for r in self.ranges]
        off_path = False
        for step in target.path:
            if step == '/':
                state = state.deal(target.board[len(state.board)])
                ranges = [remove_blocked(r, state.board) for r in ranges]
                continue
            actions = state.actions(self.settings)
            index = next(i for i, a in enumerate(actions) if a.key == step)
            actor = state.actor
            policy = self._training_policy(state, ranges[actor])
            updated = {h: w for h, w in ranges[actor].items() if policy[h] == index}
            if updated:
                ranges[actor] = updated
            else:
                # Traversals test deviations even when a deterministic action has
                # zero reach for every hand. Beliefs there are undefined: retain
                # the preceding range for this counterfactual only and disclose it.
                off_path = True
            state = state.apply(actions[index])
        return tuple(ranges), off_path

    def _river_value(self, request):
        state = request.state
        key = (state.path, state.board)
        result = self._river_cache.get(key)
        if result is None:
            if self._river_job is None:
                ranges, off_path = self._river_ranges(state)
                self._river_job = RiverJob(state, ranges, self.settings, executor=self._river_executor)
                self._river_job_key = key
                self.river_off_path_solves += int(off_path)
            if key != self._river_job_key:
                raise RuntimeError('A river traversal resumed with mismatched public state.')
            self.river_active = True
            while result is None:
                result = self._river_job.poll()
                if result is None:
                    yield
            self.river_active = False
            self._river_job = self._river_job_key = None
            self.river_solves += 1
            self.river_seconds += result.stats['total_seconds']
            self.last_river_certificate = result.stats
            self._river_cache[key] = result
            while len(self._river_cache) > self.settings.river_cache_size:
                self._river_cache.popitem(last=False)
        else:
            self.river_cache_hits += 1
            self._river_cache.move_to_end(key)
        return result.value_for_hand(request.player, request.hand)

    def _inspection_result(self, state):
        for result in reversed(self._inspection_rivers.values()):
            if state.board == result.root.board and state.path in result.nodes:
                return result
        return None

    def prepare_decision(self, state, ranges):
        if state.street != 2 or state.pending or state.terminal or any(not r for r in ranges):
            return
        key = (state.path, state.board)
        if self._inspection_result(state) is None:
            from .exact_river import ExactRiver
            result = ExactRiver(state, ranges, self.settings).solve(
                time_limit_seconds=self.settings.river_seconds, tolerance=self.settings.river_tolerance).compact()
            self._inspection_rivers[key] = result
            while len(self._inspection_rivers) > self.settings.river_cache_size:
                self._inspection_rivers.popitem(last=False)

    def _new_network(self, seed):
        # Initialize on CPU with an isolated RNG; no dropout/GPU randomness.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = Network(self.settings.deep_hidden)
        return model.to(self.device)

    def _fit(self, memory, iteration, player, strategy=False):
        if not memory.size:
            return None
        model = self._new_network(self.settings.seed + 1000 * iteration + 10 * player + int(strategy))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        label = f"{'average' if strategy else 'advantage'}_{player}"
        for step in range(self.settings.deep_train_steps):
            indices = self.batch_rng.integers(memory.size, size=self.settings.deep_batch_size)
            tensors = [torch.from_numpy(a[indices]).to(self.device) for a in
                       (memory.features, memory.targets, memory.masks, memory.rounds)]
            features, targets, masks, rounds = tensors
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(features), targets, masks, rounds, iteration, strategy)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Neural training produced a non-finite loss; the last complete policy is retained.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            self.losses[label] = float(loss.detach().cpu())
            self.phase_step = step + 1
            yield
        model.eval()
        return model

    def _rounds(self):
        while True:
            iteration = self.iterations + 1
            self.attempted_iterations += 1
            for player in (0, 1):
                self.phase, self.phase_step, self.phase_total = f"sampling_{player}", 0, self.settings.deep_traversals
                yield from self._collect(player, iteration)
                self.phase, self.phase_step, self.phase_total = f"advantage_{player}", 0, self.settings.deep_train_steps
                model = yield from self._fit(self.advantage_memory[player], iteration, player)
                self.advantages[player] = model
            averages = []
            for player in (0, 1):
                self.phase, self.phase_step, self.phase_total = f"average_{player}", 0, self.settings.deep_train_steps
                averages.append((yield from self._fit(self.strategy_memory[player], iteration, player, strategy=True)))
            self.average_models = averages
            self._policy_cache.clear()
            self._inspection_rivers.clear()
            self.iterations += 1
            self.phase, self.phase_step, self.phase_total = "round_complete", 0, 0
            yield

    def train(self, iterations=None, progress=None, *, time_limit_seconds=None):
        if self._failed:
            raise ValueError("This neural run failed. Start a new solve or refinement; the last complete average policy remains available.")
        if (iterations is None) == (time_limit_seconds is None):
            raise ValueError("Choose either rounds or a time limit.")
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError("Rounds must be a positive integer.")
        if time_limit_seconds is not None and (type(time_limit_seconds) not in (int, float) or
                                               not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError("Time limit must be positive and finite.")
        start, previous = time.monotonic(), self.elapsed
        deadline = start + time_limit_seconds if time_limit_seconds is not None else None
        target = self.iterations + iterations if iterations is not None else None
        self.time_limit_seconds, self.run_elapsed, self.status = time_limit_seconds, 0.0, "running"
        last_report = start
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        if self._work is None:
            self._work = self._rounds()
        try:
            while target is None or self.iterations < target:
                if self.cancel.is_set():
                    self.status = "cancelled"
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                next(self._work)
                now = time.monotonic()
                if progress and now - last_report >= .5:
                    self.run_elapsed, self.elapsed = now - start, previous + now - start
                    progress(self.stats(target))
                    last_report = now
            if self.status == "running":
                self.status = "complete"
        except Exception:
            self.status = "error"
            self._failed = True
            raise
        finally:
            if self._river_job is not None:
                self._river_job.pause()
            self._river_executor.pause()
            self.run_elapsed = time.monotonic() - start
            self.elapsed = previous + self.run_elapsed
            torch.set_num_threads(previous_threads)
        return self.stats(target)

    def stats(self, target=None):
        memories = self.advantage_memory + self.strategy_memory
        return {"status": self.status, "iterations": self.iterations, "target": target,
                "nodes": len(self.nodes), "seconds": round(self.elapsed, 2),
                "run_seconds": round(self.run_elapsed, 2), "time_limit_seconds": self.time_limit_seconds,
                "attempted_iterations": self.attempted_iterations,
                "completed_traversals": list(self.completed_traversals), "tree_visits": self._visits,
                "exploitability": None, "backend": "deep_cfr", "device": str(self.device),
                "policy_update": "alternating_deterministic", "iteration_unit": "neural rounds",
                "phase": "river_lp" if self.river_active else self.phase, "phase_step": self.phase_step, "phase_total": self.phase_total,
                "pending_round": self.attempted_iterations > self.iterations,
                "prediction_ready": all(m is not None for m in self.average_models),
                "tracking_limit": TRACKING_LIMIT, "tracking_saturated": self.tracking_saturated,
                "replay_capacity_bytes": sum(m.nbytes for m in memories),
                "replay": [{"size": m.size, "capacity": m.capacity, "seen": m.seen} for m in memories],
                "losses": dict(self.losses), "policy_mode": "deterministic_flop_turn",
                "river_method": "sequence_form_lp", "river_solves": self.river_solves,
                "river_cache_hits": self.river_cache_hits, "river_seconds": self.river_seconds,
                "river_off_path_solves": self.river_off_path_solves,
                "last_river_certificate": self.last_river_certificate}

    def _predict(self, state, hands):
        """Batch inspection/range propagation too, not just training traversal."""
        if state.street == 2:
            return
        actions = state.actions(self.settings)
        model = self.average_models[state.actor]
        if model is None or not actions:
            return
        missing = [h for h in hands if self.info_key(state, h) not in self._policy_cache]
        for start in range(0, len(missing), 256):
            batch = missing[start:start + 256]
            features = np.stack([self._encode(state, h, actions) for h in batch])
            with torch.inference_mode():
                logits = model(torch.from_numpy(features).to(self.device))[:, :len(actions)]
                probabilities = torch.nn.functional.one_hot(logits.argmax(dim=1), num_classes=len(actions)).cpu().numpy()
            if len(self._policy_cache) + len(batch) > 4096:
                self._policy_cache.clear()
            for hand, probs in zip(batch, probabilities):
                self._policy_cache[self.info_key(state, hand)] = probs.tolist()

    def policy(self, state, hand):
        if state.street == 2:
            result = self._inspection_result(state)
            return result.policy(state, hand) if result else (None, 0, 0)
        self._predict(state, [hand])
        key = self.info_key(state, hand)
        counts = self.nodes.get(key, SampleCounts())
        return self._policy_cache.get(key), counts.updates, counts.averages

    def prepare_policies(self, state, hands):
        self._policy_cache.clear()
        self._predict(state, list(hands))

    def inspect(self, path, selected=None):
        response = super().inspect(path, selected)
        state, ranges, _ = self.locate(path)
        if state.street == 2:
            result = self._inspection_result(state)
            response['unreachable'] = any(not r for r in ranges)
            return exact_view(response, result) if result else response
        response["prediction_coverage"] = response["coverage"]
        response["coverage"] = sum(row["mass"] for row in response["rows"] if row["updates"] and row["averages"])
        response["neural"] = {"predictions": True, "tracking_saturated": self.tracking_saturated,
                              "notice": "Network predictions include unseen hands. Direct sample counts are not prediction confidence."}
        for row in response["rows"]:
            row["policy_source"] = "neural_average" if row["probabilities"] is not None else "unavailable"
        if response["selected"] is not None:
            response["selected"]["policy_source"] = "neural_average" if response["selected"]["probabilities"] is not None else "unavailable"
        return response

    def coverage_diagnostics(self):
        from .diagnostics import sample_summary
        from .game import STREETS
        streets = {name: sample_summary(
            {"updates": counts.updates, "averages": counts.averages}
            for key, counts in self.nodes.items() if len(key[3]) - 3 == street)
            for street, name in enumerate(STREETS)}
        return {"streets": streets,
                "scope": "bounded exact-state visit counters; may exclude visited states after tracking saturates",
                "tracking_saturated": self.tracking_saturated,
                "notice": "Predictions do not count as direct sampling or verified convergence."}
