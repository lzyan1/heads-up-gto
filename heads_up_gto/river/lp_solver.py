"""Direct river study adapter; no neural fit or regret matching on this street."""
import math
import threading
import time
from ..range_game import RangeGame
from .lp_jobs import RiverJob


def exact_view(response, result):
    response.pop('training', None)
    response['exact'] = {'method': 'sequence_form_lp', 'certificate': result.stats}
    for row in response['rows']:
        row.update(updates=None, averages=None, policy_source='exact_river')
    if response['selected'] is not None:
        response['selected'].update(updates=None, averages=None, policy_source='exact_river')
    return response


class RiverOnlySolver(RangeGame):
    def __init__(self, root, ranges, settings):
        if root.street != 2 or len(root.board) != 5 or root.terminal or root.pending:
            raise ValueError('Direct solving requires an active river decision.')
        super().__init__(root, ranges, settings)
        self.result = None
        self.job = RiverJob(root, ranges, settings)
        self.cancel = threading.Event()
        self.iterations = 0
        self.status = 'ready'
        self.elapsed = self.run_elapsed = 0.
        self.time_limit_seconds = None

    def train(self, iterations=None, progress=None, *, time_limit_seconds=None):
        if self.result is not None:
            return self.stats()
        if (iterations is None) == (time_limit_seconds is None):
            raise ValueError('Choose a run budget.')
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError('Iterations must be positive.')
        if time_limit_seconds is not None and (type(time_limit_seconds) not in (int, float) or
                not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError('Time limit must be positive and finite.')
        start, previous = time.monotonic(), self.elapsed
        self.status, self.time_limit_seconds = 'running', time_limit_seconds
        report = start
        try:
            while True:
                if self.cancel.is_set():
                    self.status = 'cancelled'
                    break
                if time_limit_seconds is not None and time.monotonic() - start >= time_limit_seconds:
                    self.status = 'paused'
                    break
                self.result = self.job.poll()
                if self.result is not None:
                    self.iterations, self.status = 1, 'complete'
                    break
                now = time.monotonic()
                if progress and now - report >= .5:
                    self.run_elapsed, self.elapsed = now - start, previous + now - start
                    progress(self.stats())
                    report = now
        except Exception:
            self.status = 'error'
            raise
        finally:
            self.job.pause()
            self.run_elapsed = time.monotonic() - start
            self.elapsed = previous + self.run_elapsed
        return self.stats()

    def stats(self, target=None):
        return dict(status=self.status, backend='exact_river', iterations=self.iterations,
                    target=1, seconds=round(self.elapsed, 2), run_seconds=round(self.run_elapsed, 2),
                    time_limit_seconds=self.time_limit_seconds, prediction_ready=self.result is not None,
                    phase='river_lp', pending_round=False,
                    certificate=self.result.stats if self.result else None)

    def prepare_policies(self, state, hands):
        pass

    def policy(self, state, hand):
        return self.result.policy(state, hand) if self.result else (None, 0, 0)

    def inspect(self, path, selected=None):
        response = super().inspect(path, selected)
        return exact_view(response, self.result) if self.result else response

    def coverage_diagnostics(self):
        return {'method': 'exact_river', 'certificate': self.result.stats if self.result else None}
