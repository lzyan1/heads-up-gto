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
        self.assertEqual(preview['range_approximation']['frequency_step'],.1)

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
        self.assertTrue(self.request('/api/settings/validate', {'settings': {'buckets': 8}})['ok'])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/settings/validate', {'settings': {'buckets': 0}})
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
        self.assertAlmostEqual(bb_response['coverage'], 1)
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

    def test_solve_navigate_refine_export_and_resume(self):
        payload={'flop':'2c 7d 9h','preflop':'raise:2.5/call','iterations':100,'settings':{'sizes':[[.5],[.5],[.5]],'include_all_in':False,'max_raises':0,'equity_samples':8,'buckets':4}}
        job=self.wait(self.request('/api/solve',payload));self.assertEqual(job['stats']['status'],'complete')
        prefix='/api/jobs/'+job['id']
        diagnostic=self.request(prefix+'/diagnostics',{})
        self.assertEqual(sum(x['hands'] for x in diagnostic['coverage']['streets'].values()),job['stats']['nodes'])
        self.assertEqual(diagnostic['stats']['completed_traversals'],[100,100])
        root=self.request(prefix+'/node',{'path':[],'hand':'As8s'})
        self.assertIn('training',root)
        self.assertEqual(root['actor'],0);self.assertAlmostEqual(sum(root['aggregate']),1)
        self.assertIn('bet:97.5', [a['key'] for a in root['actions']])
        facing_jam=self.request(prefix+'/node',{'path':['bet:97.5']})
        self.assertEqual([a['key'] for a in facing_jam['actions']], ['fold', 'call'])
        self.assertEqual(self.request(prefix+'/node',{'path':['bet:97.5','call']})['terminal'], 'showdown')
        pending=self.request(prefix+'/node',{'path':['check','check']});self.assertTrue(pending['pending'])
        turn=self.request(prefix+'/node',{'path':['check','check','deal:Ts']});self.assertEqual(turn['street'],'Turn')
        self.assertIn('bet:97.5', [a['key'] for a in turn['actions']])
        refined=self.wait(self.request(prefix+'/refine',{'path':['check','check','deal:Ts'],'iterations':100}))
        self.assertTrue(refined['metadata']['local_refinement'])
        newprefix='/api/jobs/'+refined['id']
        node=self.request(newprefix+'/node',{'path':[],'hand':'As8s'});self.assertEqual(node['street'],'Turn')
        river=self.request(newprefix+'/node',{'path':['check','check','deal:3d']});self.assertEqual(river['street'],'River')
        self.assertIn('bet:97.5', [a['key'] for a in river['actions']])
        report=self.request(newprefix+'/export',{'path':[]})
        self.assertEqual(len(report['root_state']['board']),4);self.assertEqual(len(report['root_ranges']),2)
        resumed=self.wait(self.request(newprefix+'/resume',{'iterations':50}));self.assertEqual(resumed['stats']['iterations'],150)

    def test_stop_returns_partial_results(self):
        job=self.request('/api/solve',{'flop':'As Kd 7c','preflop':'raise:2.5/call','iterations':100000})
        self.request('/api/jobs/'+job['id']+'/stop',{})
        result=self.wait(job)
        self.assertEqual(result['stats']['status'],'cancelled')

    def test_app_training_is_uncapped_by_default(self):
        payload = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call', 'iterations': 10,
                   'settings': {'equity_samples': 8, 'buckets': 4}}
        job = self.wait(self.request('/api/solve', payload))
        self.assertEqual(job['stats']['status'], 'complete')
        self.assertGreater(job['stats']['nodes'], 1)
        self.assertIsNone(job['metadata']['settings']['max_nodes'])
        prefix = '/api/jobs/' + job['id']
        resumed = self.wait(self.request(prefix + '/resume', {'iterations': 10}))
        self.assertEqual(resumed['stats']['iterations'], 20)
        self.assertEqual(resumed['stats']['status'], 'complete')
        refined = self.wait(self.request(prefix + '/refine', {'path': ['check'], 'iterations': 10}))
        self.assertEqual(refined['stats']['status'], 'complete')
        self.assertIsNone(refined['metadata']['settings']['max_nodes'])
        self.assertTrue(self.request('/api/settings/validate', {'iterations': 200001})['ok'])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/settings/validate', {'iterations': 9})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_restored_card_settings_and_optional_decision_limit(self):
        settings = {'mode': 'exact', 'buckets': 16, 'equity_samples': 32, 'seed': 91, 'max_nodes': 1}
        payload = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call', 'iterations': 10, 'settings': settings}
        job = self.wait(self.request('/api/solve', payload))
        for key, value in settings.items():
            self.assertEqual(job['metadata']['settings'][key], value)
        self.assertEqual(job['stats']['status'], 'node_limit')
        resumed = self.wait(self.request('/api/jobs/' + job['id'] + '/resume', {'iterations': 10, 'max_nodes': None}))
        self.assertEqual(resumed['stats']['status'], 'complete')
        self.assertGreater(resumed['stats']['nodes'], 1)
        self.assertIsNone(resumed['metadata']['settings']['max_nodes'])
        self.assertEqual(resumed['metadata']['settings']['mode'], 'exact')

    def test_timed_solve_resume_refine_and_switch_back_to_iterations(self):
        payload = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call', 'time_limit_seconds': .1,
                   'settings': {'mode': 'exact'}}
        job = self.wait(self.request('/api/solve', payload))
        self.assertEqual(job['stats']['status'], 'complete')
        self.assertGreater(job['stats']['iterations'], 0)
        self.assertIsNone(job['stats']['target'])
        self.assertEqual(job['stats']['time_limit_seconds'], .1)
        self.assertGreaterEqual(job['stats']['run_seconds'], .1)
        self.assertLess(job['stats']['run_seconds'], 2)
        prefix = '/api/jobs/' + job['id']
        resumed = self.wait(self.request(prefix + '/resume', {'time_limit_seconds': .1}))
        self.assertGreater(resumed['stats']['iterations'], job['stats']['iterations'])
        self.assertGreaterEqual(resumed['stats']['seconds'], job['stats']['seconds'] + .09)
        self.assertLess(resumed['stats']['run_seconds'], 2)
        counted = self.wait(self.request(prefix + '/resume', {'iterations': 10}))
        self.assertEqual(counted['stats']['iterations'], resumed['stats']['iterations'] + 10)
        self.assertIsNone(counted['stats']['time_limit_seconds'])
        refined = self.wait(self.request(prefix + '/refine', {'path': ['check'], 'time_limit_seconds': .1,
                                                          'settings': {'mode': 'bucketed', 'buckets': 6, 'equity_samples': 8, 'seed': 23}}))
        self.assertEqual(refined['stats']['status'], 'complete')
        self.assertEqual(refined['metadata']['settings']['buckets'], 6)
        self.assertEqual(refined['metadata']['training_budget'], {'time_limit_seconds': .1})
        report = self.request('/api/jobs/' + refined['id'] + '/export', {'path': []})
        self.assertEqual(report['stats']['time_limit_seconds'], .1)
        self.assertEqual(report['metadata']['settings']['seed'], 23)

    def test_invalid_time_limits_rejected_on_every_training_endpoint(self):
        base = {'flop': 'As Kd 7c', 'preflop': 'raise:2.5/call'}
        job = self.wait(self.request('/api/solve', {**base, 'iterations': 10, 'settings': {'mode': 'exact'}}))
        prefix = '/api/jobs/' + job['id']
        endpoints = ('/api/settings/validate', '/api/solve', prefix + '/resume', prefix + '/refine')
        for budget in ({'time_limit_seconds': 0}, {'time_limit_seconds': -1}, {'time_limit_seconds': '2'}, {'time_limit_seconds': True}, {'time_limit_seconds': None}, {'time_limit_seconds': float('inf')}, {'time_limit_seconds': float('nan')}, {'time_limit_seconds': 1, 'iterations': 10}):
            for endpoint in endpoints:
                with self.subTest(budget=budget, endpoint=endpoint), self.assertRaises(HTTPError) as error:
                    self.request(endpoint, {**base, **budget, 'path': []})
                self.assertEqual(error.exception.code, 400)
                error.exception.close()

    def test_refine_keeps_sizing_context(self):
        job = self.wait(self.request('/api/solve', {'flop': 'As Td 6h', 'preflop': 'raise:2.5/call', 'iterations': 10, 'settings': {'equity_samples': 8, 'buckets': 4}}))
        prefix = '/api/jobs/' + job['id']
        path = ['check', 'bet:1.7', 'raise:8']
        before = self.request(prefix + '/node', {'path': path})
        self.assertEqual([a['key'] for a in before['actions']], ['fold', 'call', 'raise:97.5'])
        refined = self.wait(self.request(prefix + '/refine', {'path': path, 'iterations': 10, 'settings': {'sizes': [[3], [.6], [.9]], 'include_all_in': False, 'max_raises': 0, 'equity_samples': 8, 'buckets': 4}}))
        prefix = '/api/jobs/' + refined['id']
        after = self.request(prefix + '/node', {'path': []})
        self.assertEqual(before['actions'], after['actions'])
        self.assertEqual(before['sizing'], after['sizing'])
        jam = self.request(prefix + '/node', {'path': ['raise:97.5']})
        self.assertEqual([a['key'] for a in jam['actions']], ['fold', 'call'])
        turn_path = ['call', 'deal:2c', 'check']
        turn = self.request(prefix + '/node', {'path': turn_path})
        self.assertEqual(turn['sizing']['scenario'], 'stab')
        second = self.wait(self.request(prefix + '/refine', {'path': turn_path, 'iterations': 10}))
        report = self.request('/api/jobs/' + second['id'] + '/export', {'path': []})
        self.assertEqual(turn['actions'], report['node']['actions'])
        self.assertEqual(report['root_state']['aggressor_history'], [0])
        self.assertEqual(report['root_state']['path'], [])
        self.assertEqual(report['metadata']['sizing']['profile'], 'single_raised')
        self.assertEqual(report['metadata']['settings']['sizes'], [[], [], []])


if __name__=='__main__':unittest.main()
