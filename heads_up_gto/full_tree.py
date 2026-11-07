"""Study-app adapter for the full-width postflop solver (flop, turn or river root).

The app's server drives solvers through a small interface: ``train``, ``stats``, ``cancel``,
``locate``, ``inspect`` and ``coverage_diagnostics``. ``RangeGame`` already supplies navigation
(``locate``, ``timeline``, ``inspect``) on top of ``policy(state, hand)``; this class serves that
policy from ``PostflopCFR``'s average strategy, so every street (including mixed flop and turn
strategies) comes from one solve of the whole remaining game, with no neural networks.
"""
import math
import os
import threading
import time

from .cards import card_text
from .postflop_cfr import PostflopCFR
from .range_game import RangeGame

METHOD = 'full_tree_dcfr'


def memory_limit():
    """Bytes a solve may use: 60% of physical memory (8 GiB if it cannot be read)."""
    try:
        return int(.6 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        return 8 * 2 ** 30


class FullTreeSolver(RangeGame):
    def __init__(self, root, ranges, settings, enforce_memory=True):
        super().__init__(root, ranges, settings)
        # Lazy: builds the public tree and checks the memory estimate (raising ValueError with a
        # readable message) without allocating, so previews and rejected solves are cheap.
        self.engine = PostflopCFR(root, ranges, settings, lazy=True,
                                  max_bytes=memory_limit() if enforce_memory else float('inf'))
        self.cancel = threading.Event()
        self.iterations = 0
        self.status = 'ready'
        self.phase = 'solving'
        self.elapsed = self.run_elapsed = 0.
        self.time_limit_seconds = None
        self._cache = None

    @property
    def memory_bytes(self):
        return self.engine.memory_bytes

    # -------------------------------------------------------------- training
    def train(self, iterations=None, progress=None, *, time_limit_seconds=None):
        if (iterations is None) == (time_limit_seconds is None):
            raise ValueError('Choose either iterations or a time limit.')
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError('Iterations must be a positive integer.')
        if time_limit_seconds is not None and (type(time_limit_seconds) not in (int, float) or
                                               not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError('Time limit must be positive and finite.')
        start, previous = time.monotonic(), self.elapsed
        target = self.iterations + iterations if iterations is not None else None
        self.time_limit_seconds, self.run_elapsed, self.status = time_limit_seconds, 0., 'running'
        state = {'reported': start}

        def clock():
            self.run_elapsed = time.monotonic() - start
            self.elapsed = previous + self.run_elapsed

        def stop():
            return self.cancel.is_set() or (time_limit_seconds is not None
                                            and time.monotonic() - start >= time_limit_seconds)

        def tick(iteration, _seconds, _exploit):
            self.iterations = iteration
            now = time.monotonic()
            if progress and now - state['reported'] >= .5:
                clock()
                progress(self.stats(target))
                state['reported'] = now

        try:
            if not self.engine.allocated and not self.cancel.is_set():
                self.phase = 'preparing'
                clock()
                if progress:
                    progress(self.stats(target))
                self.engine.allocate()
            self.phase = 'solving'
            if not self.cancel.is_set():
                self.engine.solve(iterations=iterations if iterations is not None else 2 ** 62, stop=stop,
                                  progress=tick, check_every=2 ** 62, check_overhead=.15, final_check=False)
            self.iterations = self.engine.iterations
            self.status = 'cancelled' if self.cancel.is_set() else 'complete'
            # A closing check unless stopped by the user: stop must stay responsive, and the last
            # periodic check (labelled with its iteration) is reported instead.
            if self.status == 'complete' and self.iterations and (
                    self.engine.last_check is None or self.engine.last_check[0] != self.iterations):
                self.phase = 'checking'
                clock()
                if progress:
                    progress(self.stats(target))
                self.engine.check()
        except Exception:
            self.status = 'error'
            raise
        finally:
            self.phase = 'solving'
            clock()
        return self.stats(target)

    # ----------------------------------------------------------------- stats
    def certificate(self):
        check = self.engine.last_check
        if check is None:
            return None
        iteration, exploit, value, best = check
        return {'method': METHOD, 'exploitability_bb': float(exploit),
                'exploitability_pot': float(exploit / self.root.pot), 'value_bb': float(value),
                'best_response_values_bb': [float(b) for b in best], 'iterations_at_check': iteration,
                'current': iteration == self.iterations, 'engine': self.engine.engine,
                'pot_bb': self.root.pot}

    def stats(self, target=None):
        certificate = self.certificate()
        return dict(status=self.status, backend='full_tree', iterations=self.iterations, target=target,
                    seconds=round(self.elapsed, 2), run_seconds=round(self.run_elapsed, 2),
                    time_limit_seconds=self.time_limit_seconds, prediction_ready=self.iterations > 0,
                    phase=self.phase, pending_round=False, iteration_unit='iterations',
                    exploitability_bb=certificate['exploitability_bb'] if certificate else None,
                    exploitability_pot=certificate['exploitability_pot'] if certificate else None,
                    certificate=certificate, engine=self.engine.engine,
                    memory_gib=round(self.engine.memory_bytes / 2 ** 30, 2), hands=self.engine.size,
                    decisions=self.engine.stats['public_decisions'], contexts=self.engine.stats['max_contexts'])

    def coverage_diagnostics(self):
        return {'method': METHOD, 'certificate': self.certificate(), 'engine': self.engine.engine,
                'memory_bytes': self.engine.memory_bytes}

    # ---------------------------------------------------------------- policy
    def _tokens(self, state):
        """Path tokens for ``PostflopCFR``: action keys, with each street change as the dealt card."""
        base, dealt, out = len(self.root.board), 0, []
        for step in state.path[len(self.root.path):]:
            if step == '/':
                out.append(card_text(state.board[base + dealt]))
                dealt += 1
            else:
                out.append(step)
        return out

    def prepare_policies(self, state, hands):
        self._cache = None
        if not self.engine.allocated or not self.iterations:
            return
        key = (tuple(self._tokens(state)), state.board)
        actions, probabilities = self.engine.strategy(list(key[0]))
        self._cache = (key, probabilities, {h: i for i, h in enumerate(self.engine.hands)})

    def policy(self, state, hand):
        if not self.engine.allocated or not self.iterations:
            return None, 0, 0
        key = (tuple(self._tokens(state)), state.board)
        if self._cache is None or self._cache[0] != key:
            self.prepare_policies(state, None)
        _, probabilities, index = self._cache
        return probabilities[:, index[tuple(sorted(hand))]].tolist(), 0, 0

    def inspect(self, path, selected=None):
        response = super().inspect(path, selected)
        response.pop('training', None)
        certificate = self.certificate()
        response['exact'] = {'method': METHOD, 'certificate': certificate or {}}
        for row in response['rows']:
            row.update(updates=None, averages=None, policy_source=METHOD)
        if response['selected'] is not None:
            response['selected'].update(updates=None, averages=None, policy_source=METHOD)
        return response
