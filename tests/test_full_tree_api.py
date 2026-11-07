"""Full-tree solver behind the web API: solve, navigate, refine, resume, stop, export, limits."""
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError

from heads_up_gto.server import Handler, JOBS, prepare
import tests.test_api as base

SPOT = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/raise:11.5/raise:26/call'}  # 4-bet pot: a small tree


class FullTreeAPITests(unittest.TestCase):
    request, wait = base.APITests.request, base.APITests.wait

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        JOBS.clear()

    def solve(self, **changes):
        return self.wait(self.request('/api/solve', {**SPOT, 'iterations': 40, **changes}))

    def test_solve_inspect_and_navigate_every_street(self):
        job = self.solve()
        prefix = '/api/jobs/' + job['id']
        stats = job['stats']
        self.assertEqual((stats['backend'], stats['status'], stats['iterations']), ('full_tree', 'complete', 40))
        self.assertEqual(job['metadata']['solver'], 'full_tree')
        self.assertLess(stats['exploitability_pot'], .05)
        self.assertTrue(stats['certificate']['current'])
        root = self.request(prefix + '/node', {'path': []})
        self.assertEqual(root['street'], 'Flop')
        self.assertAlmostEqual(sum(root['aggregate']), 1, places=5)
        self.assertAlmostEqual(root['coverage'], 1)
        self.assertEqual(root['exact']['method'], 'full_tree_dcfr')
        self.assertNotIn('training', root)
        for row in root['rows']:
            self.assertAlmostEqual(sum(row['probabilities']), 1, places=5)
            self.assertIsNone(row['updates'])
        # Flop strategies are mixed, not one deterministic action per hand.
        self.assertTrue(any(max(r['probabilities']) < .999 for r in root['rows']))
        turn = self.request(prefix + '/node', {'path': ['check', 'check', 'deal:2h']})
        self.assertEqual((turn['street'], turn['board'][-1]), ('Turn', '2h'))
        self.assertAlmostEqual(sum(turn['aggregate']), 1, places=5)
        river = self.request(prefix + '/node', {'path': ['check', 'check', 'deal:2h', 'check', 'check', 'deal:9s']})
        self.assertEqual(river['street'], 'River')
        self.assertAlmostEqual(sum(river['aggregate']), 1, places=5)
        self.assertEqual(len(river['timeline']), 7)
        picked = self.request(prefix + '/node', {'path': [], 'hand': root['rows'][0]['hand']})
        self.assertEqual(picked['selected']['hand'], root['rows'][0]['hand'])
        # Dealing a card that blocks hands removes them from the displayed range.
        ace = self.request(prefix + '/node', {'path': ['check', 'check', 'deal:Ah']})
        blocked = {r['hand'] for r in root['rows'] if 'Ah' in (r['hand'][:2], r['hand'][2:])}
        self.assertTrue(blocked)
        self.assertFalse(blocked & {r['hand'] for r in ace['rows']})

    def test_refine_resume_diagnostics_and_export(self):
        job = self.solve()
        prefix = '/api/jobs/' + job['id']
        path = ['check', 'check', 'deal:2h']
        before = self.request(prefix + '/node', {'path': path})
        refined = self.wait(self.request(prefix + '/refine', {'path': path, 'iterations': 60}))
        self.assertEqual(refined['stats']['backend'], 'full_tree')
        self.assertTrue(refined['metadata']['local_refinement'])
        self.assertEqual(refined['stats']['iterations'], 60)
        node = self.request('/api/jobs/' + refined['id'] + '/node', {'path': []})
        self.assertEqual(node['street'], 'Turn')
        self.assertEqual(node['actions'], before['actions'])
        self.assertEqual(node['sizing'], before['sizing'])
        resumed = self.wait(self.request('/api/jobs/' + refined['id'] + '/resume', {'iterations': 10}))
        self.assertEqual(resumed['stats']['iterations'], 70)
        diagnostic = self.request('/api/jobs/' + refined['id'] + '/diagnostics', {})
        self.assertEqual(diagnostic['coverage']['method'], 'full_tree_dcfr')
        exported = self.request('/api/jobs/' + refined['id'] + '/export', {'path': []})
        self.assertEqual(exported['metadata']['solver'], 'full_tree')
        self.assertIn('Approximate equilibrium', exported['notice'])
        self.assertEqual(len(exported['root_state']['board']), 4)

    def test_stop_cancels_and_resume_continues(self):
        job = self.request('/api/solve', {**SPOT, 'iterations': 10 ** 9})
        prefix = '/api/jobs/' + job['id']
        with self.assertRaises(HTTPError) as error:
            self.request(prefix + '/node', {})
        self.assertEqual(error.exception.code, 409)
        error.exception.close()
        time.sleep(.5)
        self.request(prefix + '/stop', {})
        stopped = self.wait(job)
        self.assertEqual(stopped['stats']['status'], 'cancelled')
        self.assertGreater(stopped['stats']['iterations'], 0)
        resumed = self.wait(self.request(prefix + '/resume', {'iterations': 5}))
        self.assertEqual(resumed['stats']['iterations'], stopped['stats']['iterations'] + 5)

    def test_zero_work_budget_reports_untrained_policies(self):
        job = self.wait(self.request('/api/solve', {**SPOT, 'time_limit_seconds': 1e-9}))
        self.assertEqual(job['stats']['iterations'], 0)
        root = self.request('/api/jobs/' + job['id'] + '/node', {'path': []})
        self.assertTrue(all(r['probabilities'] is None for r in root['rows']))
        self.assertEqual(root['aggregate'], [])

    def test_memory_limit_rejects_solves_but_not_previews(self):
        with mock.patch('heads_up_gto.full_tree.memory_limit', return_value=1000):
            with self.assertRaises(HTTPError) as error:
                self.request('/api/solve', {**SPOT, 'iterations': 1})
            self.assertEqual(error.exception.code, 400)
            self.assertIn('needs about', error.exception.read().decode())
            error.exception.close()
            self.assertEqual(self.request('/api/preview', SPOT)['backend'], 'full_tree')

    def test_solver_choice_is_validated_and_legacy_is_still_available(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/api/settings/validate', {'solver': 'magic'})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        self.assertTrue(self.request('/api/settings/validate', {'solver': 'legacy'})['ok'])
        legacy = prepare({**SPOT, 'solver': 'legacy', 'iterations': 1}, preview=True)
        self.assertEqual(legacy[2]['backend'], 'hybrid_pure')

    def test_older_solves_are_evicted_to_make_room(self):
        first = self.solve(iterations=5)
        with mock.patch('heads_up_gto.server.memory_limit', return_value=int(1.5 * 0.18 * 2 ** 30)):
            second = self.solve(iterations=5)
        self.assertNotIn(first['id'], JOBS)
        self.assertIn(second['id'], JOBS)


if __name__ == '__main__':
    unittest.main()
