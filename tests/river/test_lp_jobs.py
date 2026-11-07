import unittest
from heads_up_gto.cards import parse_cards,parse_range
from heads_up_gto.game import Settings,State
from heads_up_gto.river.lp_jobs import RiverExecutor,RiverJob
from heads_up_gto.river.lp_solver import RiverOnlySolver


class RiverWorkerTests(unittest.TestCase):
    def inputs(self):
        root=State(2,(1,1),parse_cards('2c 5d 7h 9s Jc'),street=2)
        r=parse_range('As3s,Kh3h,Qd3d')
        return root,(r,r),Settings(sizes=((),(),(.5,)),max_raises=0,river_seconds=10)

    def test_process_is_reused_for_completed_boundaries_and_closed_on_pause(self):
        executor=RiverExecutor()
        try:
            pids=[]
            for _ in range(2):
                job=RiverJob(*self.inputs(),executor=executor)
                result=None
                while result is None:
                    result=job.poll()
                pids.append(executor.process.pid)
                self.assertAlmostEqual(result.stats['value_bb'],-1/18)
            self.assertEqual(*pids)
        finally:
            executor.pause()
        self.assertIsNone(executor.process)

    def test_timed_pause_interrupts_worker_and_resume_finishes_same_river(self):
        solver=RiverOnlySolver(*self.inputs())
        paused=solver.train(time_limit_seconds=.03)
        self.assertEqual(paused['status'],'paused')
        self.assertIsNone(solver.job.executor.process)
        result=solver.train(iterations=1)
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['certificate']['value_bb'],-1/18)
        self.assertIsNone(solver.job.executor.process)
        self.assertIn('exact',solver.inspect([]))
