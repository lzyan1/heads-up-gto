"""Cancellable CPU river LP workers, shared by training boundaries and direct study."""
import multiprocessing


def _solve_loop(connection):
    from .exact_lp import ExactRiver
    try:
        while True:
            try:
                root, ranges, settings = connection.recv()
            except EOFError:
                return
            try:
                result = ExactRiver(root, ranges, settings).solve(time_limit_seconds=settings.river_seconds,
                                                                 tolerance=settings.river_tolerance)
                connection.send(('ok', result.compact()))
            except Exception as exc:
                connection.send(('error', str(exc)))
    finally:
        connection.close()


class RiverExecutor:
    """One process reused across boundaries within a training run."""
    def __init__(self):
        self.process = self.connection = None

    def submit(self, payload):
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.connection, worker = context.Pipe()
            self.process = context.Process(target=_solve_loop, args=(worker,), daemon=True)
            self.process.start()
            worker.close()
        self.connection.send(payload)

    def poll(self):
        if not self.connection.poll(.02):
            return None
        try:
            status, result = self.connection.recv()
        except EOFError as exc:
            self.pause()
            raise RuntimeError('The river worker exited without a certified solution.') from exc
        if status != 'ok':
            raise RuntimeError('Direct river solve failed: ' + result)
        return result

    def pause(self):
        # HiGHS has no optimizer checkpoint: resume restarts an interrupted LP.
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.connection.close()
            self.process = self.connection = None


class RiverJob:
    def __init__(self, root, ranges, settings, *, executor=None):
        self.payload = (root, ranges, settings)
        self.executor = executor or RiverExecutor()
        self.started = False

    def poll(self):
        if not self.started:
            self.executor.submit(self.payload)
            self.started = True
        result = self.executor.poll()
        if result is not None:
            self.started = False
        return result

    def pause(self):
        self.executor.pause()
        self.started = False
