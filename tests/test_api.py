import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from heads_up_gto.server import Handler, JOBS


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
        JOBS.clear()

    def request(self,path,data=None,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        with urlopen(req,timeout=30) as response:return json.load(response)

    def wait(self,job):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            result=self.request('/api/jobs/'+job['id'])
            if not result['busy']:return result
            time.sleep(.02)
        self.fail('Solve timed out')

    def test_health_and_page(self):
        self.assertTrue(self.request('/api/health')['ok'])
        with urlopen(self.base) as response:
            self.assertIn(b'Solve continuation',response.read())
            self.assertIn("frame-ancestors 'none'",response.headers['Content-Security-Policy'])

    def test_cross_origin_rejected(self):
        with self.assertRaises(HTTPError) as error:self.request('/api/solve',{},'https://example.com')
        self.assertEqual(error.exception.code,400);error.exception.close()

    def test_invalid_inputs(self):
        for data in ({'flop':'As As Kd','preflop':'raise:2.5/call'}, {'flop':'As Kd 7c','preflop':'raise:2.5/check'}):
            with self.assertRaises(HTTPError) as error:self.request('/api/solve',data)
            self.assertEqual(error.exception.code,400);error.exception.close()

    def test_builtin_chart_and_default_source(self):
        chart=self.request('/api/builtin-chart')
        self.assertEqual(len(chart['nodes']),6)
        self.assertFalse(chart['is_demo'])
        preview=self.request('/api/preview',{'flop':'As Kd 7c','preflop':'raise:2.5/raise:11.5/raise:26/call'})
        self.assertEqual(preview['pot'],52)
        self.assertEqual(preview['range_approximation']['frequency_step'],1.)

    def test_preflop_navigation_and_settings(self):
        root = self.request('/api/preflop', {'history': []})
        self.assertEqual(root['actor'], 1)
        self.assertEqual(len(root['timeline']), 1)
        flop = self.request('/api/preflop', {'history': ['raise:2.5', 'raise:11.5', 'call']})
        self.assertTrue(flop['pending'])
        self.assertEqual(flop['pot'], 23)
        for payload in ({'history': ['check']}, {'range_source': 'unknown'}, {'range_source': 'chart'}):
            with self.assertRaises(HTTPError) as error:
                self.request('/api/preflop', payload)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        self.assertTrue(self.request('/api/settings/validate', {'settings': {'deep_hidden': 64}})['ok'])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/settings/validate', {'settings': {'deep_hidden': 0}})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        with urlopen(self.base+'/model.mjs') as response:
            self.assertIn('text/javascript', response.headers['Content-Type'])
            self.assertIn(b'branchPath', response.read())

    def test_only_hardcoded_charts_are_accepted(self):
        valid = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call'}
        for override in ({'range_source': 'manual'}, {'range_source': 'demo'}, {'range_source': 'chart'}, {'chart': {}}, {'bb_range': 'AA'}, {'sb_range': 'KK'}):
            for endpoint in ('/api/preflop', '/api/preview', '/api/solve'):
                with self.subTest(endpoint=endpoint, override=override), self.assertRaises(HTTPError) as error:
                    self.request(endpoint, {**valid, **override})
                self.assertEqual(error.exception.code, 400)
                self.assertIn('hardcoded GTO Wizard', error.exception.read().decode())
                error.exception.close()
        preview = self.request('/api/preview', valid)
        self.assertEqual(preview['range_source'], 'wizard')
        self.assertEqual(preview['sizing_status'], dict.fromkeys(('flop', 'turn', 'river'), 'scenario_heuristic'))
        self.assertEqual(preview['settings']['sizes'], [[], [], []])

    def test_preflop_jams_start_at_sb_four_bet(self):
        prefixes = [[], ['raise:2.5']]
        for history in prefixes:
            result = self.request('/api/preflop', {'history': history})
            self.assertNotIn('raise:100', [a['key'] for a in result['actions']])
            with self.assertRaises(HTTPError) as error:
                self.request('/api/preflop', {'history': history+['raise:100']})
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        sb_four_bet = ['raise:2.5', 'raise:11.5']
        sb = self.request('/api/preflop', {'history': sb_four_bet})
        self.assertIn('raise:100', [a['key'] for a in sb['actions']])
        bb_response = self.request('/api/preflop', {'history': sb_four_bet + ['raise:100']})
        self.assertEqual(bb_response['actor'], 0)
        self.assertEqual([a['key'] for a in bb_response['actions']], ['fold', 'call'])
        self.assertFalse(bb_response.get('missing_chart'))
        self.assertEqual(bb_response['coverage'], 0)
        called = self.request('/api/preflop', {'history': sb_four_bet + ['raise:100', 'call']})
        self.assertEqual(called['terminal'], 'showdown')
        self.assertEqual(called['pot'], 200)
        self.assertFalse(called['pending'])
        folded = self.request('/api/preflop', {'history': sb_four_bet + ['raise:100', 'fold']})
        self.assertEqual(folded['terminal'], 'fold')
        self.assertEqual(folded['winner'], 1)
        history = ['raise:2.5', 'raise:11.5', 'raise:26']
        bb = self.request('/api/preflop', {'history': history})
        self.assertEqual(bb['actor'], 0)
        self.assertIn('raise:100', [a['key'] for a in bb['actions']])
        sb = self.request('/api/preflop', {'history': history+['raise:100']})
        self.assertEqual([a['key'] for a in sb['actions']], ['fold', 'call'])
        called = self.request('/api/preflop', {'history': history+['raise:100', 'call']})
        self.assertEqual(called['terminal'], 'showdown')
        self.assertEqual(called['pot'], 200)

    settings = dict(deep_hidden=32, deep_buffer_size=128, deep_traversals=2,
                    deep_train_steps=2, deep_batch_size=16, deep_inference_batch=2)

    def start(self, **changes):
        payload = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/raise:11.5/raise:26/call',
                   'time_limit_seconds': 1e-9, 'settings': self.settings}
        return self.wait(self.request('/api/solve', {**payload, **changes}))

    def test_default_solver_budget_and_removed_settings(self):
        from heads_up_gto.server import training_budget
        from heads_up_gto.deep_cfr import DeepCFRSolver
        from heads_up_gto.server import prepare
        solver, budget, metadata = prepare({'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call',
                                             'settings': self.settings})
        self.assertIsInstance(solver, DeepCFRSolver)
        self.assertEqual(budget, {'time_limit_seconds': 60})
        self.assertEqual(metadata['backend'], 'hybrid_pure')
        self.assertEqual(training_budget({'iterations': 1}), {'iterations': 1})
        for key, value in dict(backend='tabular', mode='bucketed', chance_mode='enumerate_turns',
                               max_nodes=1, buckets=8, equity_samples=24).items():
            with self.subTest(key=key), self.assertRaises(HTTPError) as error:
                self.request('/api/settings/validate', {'settings': {key: value}})
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        with urlopen(self.base) as response:
            page = response.read().decode()
        self.assertNotIn('Tabular', page)
        self.assertNotIn('id="backend"', page)
        self.assertNotIn('id="max-nodes"', page)
        self.assertIn('Neural rounds per run', page)

    def test_solve_navigate_refine_train_resume_export(self):
        job = self.start()
        prefix = '/api/jobs/' + job['id']
        self.assertEqual(job['stats']['backend'], 'deep_cfr')
        self.assertFalse(job['stats']['prediction_ready'])
        root = self.request(prefix + '/node', {'path': []})
        self.assertEqual(root['aggregate'], [])
        self.assertEqual(root['prediction_coverage'], 0)
        self.assertTrue(all(r['probabilities'] is None for r in root['rows']))
        facing = self.request(prefix + '/node', {'path': ['bet:74']})
        self.assertEqual([a['key'] for a in facing['actions']], ['fold', 'call'])
        self.assertEqual(self.request(prefix + '/node', {'path': ['bet:74', 'call']})['terminal'], 'showdown')
        path = ['check', 'check', 'deal:2h', 'check', 'check', 'deal:3d']
        before = self.request(prefix + '/node', {'path': path})
        refined = self.wait(self.request(prefix + '/refine', {'path': path, 'iterations': 1}))
        self.assertTrue(refined['metadata']['local_refinement'])
        self.assertTrue(refined['metadata']['fallback_used'])
        self.assertTrue(refined['stats']['prediction_ready'])
        self.assertEqual(refined['stats']['backend'], 'exact_river')
        self.assertLess(refined['stats']['certificate']['exploitability_bb'], 1e-7)
        newprefix = '/api/jobs/' + refined['id']
        node = self.request(newprefix + '/node', {'path': []})
        self.assertEqual(node['street'], 'River')
        self.assertEqual(node['actions'], before['actions'])
        self.assertAlmostEqual(node['coverage'], 1)
        self.assertAlmostEqual(sum(node['aggregate']), 1, places=6)
        self.assertIn('exact', node)
        resumed = self.wait(self.request(newprefix + '/resume', {'iterations': 1}))
        self.assertEqual(resumed['stats']['iterations'], 1)
        diagnostic = self.request(newprefix + '/diagnostics', {})
        self.assertEqual(diagnostic['coverage']['method'], 'exact_river')
        exported = self.request(newprefix + '/export', {'path': []})
        self.assertEqual(exported['metadata']['backend'], 'exact_river')
        self.assertIn('exact', exported['node'])
        self.assertEqual(len(exported['root_state']['board']), 5)
        self.assertEqual(len(exported['root_ranges']), 2)
        self.assertNotIn('buckets', exported['metadata']['settings'])

    def test_read_only_reference_comparison(self):
        job = self.start(preflop='raise:2.5/call')
        prefix = '/api/jobs/' + job['id']
        self.assertTrue(job['metadata']['reference_line_available'])
        report = self.request(prefix + '/comparison', {})
        self.assertFalse(report['reference_frequencies_available'])
        self.assertEqual(len(report['nodes']), 7)
        self.assertEqual(report['nodes'][-1]['model']['pot'], 21)
        self.assertEqual(report['nodes'][-1]['reference']['pot'], 41)
        self.assertEqual(report['nodes'][0]['model']['aggregate'], [])
        self.assertEqual(self.request(prefix)['stats'], job['stats'])

    def test_stop_returns_partial_round_and_resume_gets_fresh_budget(self):
        job = self.request('/api/solve', {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call',
                                        'iterations': 100000, 'settings': self.settings})
        prefix = '/api/jobs/' + job['id']
        with self.assertRaises(HTTPError) as error:
            self.request(prefix + '/node', {})
        self.assertEqual(error.exception.code, 409)
        error.exception.close()
        self.request(prefix + '/stop', {})
        stopped = self.wait(job)
        self.assertEqual(stopped['stats']['status'], 'cancelled')
        resumed = self.wait(self.request(prefix + '/resume', {'time_limit_seconds': 1e-9}))
        self.assertEqual(resumed['stats']['iterations'], stopped['stats']['iterations'])
        self.assertEqual(resumed['stats']['status'], 'complete')
        self.assertEqual(resumed['stats']['time_limit_seconds'], 1e-9)
        self.assertGreaterEqual(resumed['stats']['seconds'], stopped['stats']['seconds'])

    def test_invalid_budgets_rejected_on_every_training_endpoint(self):
        base = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call'}
        job = self.start()
        prefix = '/api/jobs/' + job['id']
        endpoints = ('/api/settings/validate', '/api/solve', prefix + '/resume', prefix + '/refine')
        for budget in ({'time_limit_seconds': 0}, {'time_limit_seconds': -1}, {'time_limit_seconds': '2'},
                       {'time_limit_seconds': True}, {'time_limit_seconds': None},
                       {'time_limit_seconds': float('inf')}, {'time_limit_seconds': float('nan')},
                       {'time_limit_seconds': 1, 'iterations': 1}, {'iterations': 0}, {'iterations': True}):
            for endpoint in endpoints:
                with self.subTest(budget=budget, endpoint=endpoint), self.assertRaises(HTTPError) as error:
                    self.request(endpoint, {**base, **budget, 'path': []})
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
        with self.assertRaises(HTTPError) as error:
            self.request(prefix + '/resume', {'max_nodes': None})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_refine_keeps_sizing_context(self):
        job = self.start(flop='As Td 6h')
        prefix = '/api/jobs/' + job['id']
        path = ['check', 'check', 'deal:2c', 'check']
        before = self.request(prefix + '/node', {'path': path})
        refined = self.wait(self.request(prefix + '/refine', {'path': path, 'time_limit_seconds': 1e-9}))
        after = self.request('/api/jobs/' + refined['id'] + '/node', {'path': []})
        self.assertEqual(before['actions'], after['actions'])
        self.assertEqual(before['sizing'], after['sizing'])
        exported = self.request('/api/jobs/' + refined['id'] + '/export', {'path': []})
        self.assertEqual(exported['root_state']['aggressor_history'], [-1])



if __name__ == '__main__':
    unittest.main()
