from unittest import TestCase

from core.basis.cycler import Cycler


class TestCycler(TestCase):

    def test_periodic_run(self):
        cycler = Cycler(7)
        run_at = []
        for t in range(100):
            with cycler(t) as _, _:
                run_at.append(t)

        # 106 will run because 106-last_run_at=98 >= period
        with cycler(106) as _, _:
            run_at.append(106)

        # 112 will not run because 112-last_run_at=106 <= period
        with cycler(112) as _, _:
            run_at.append(112)

        self.assertEqual(tuple(run_at), tuple([t for t in range(100) if t % 7 == 0] + [106]))

    def test_wrong_usage_exception(self):
        cycler = Cycler(7)
        with self.assertRaises(Exception):
            with cycler(7):
                pass
