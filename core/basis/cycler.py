class NotAtPeriod(Exception):
    pass


class Cycler(object):

    def __init__(self, period: int, start: int = 0, run_at_start=False):
        """
        Run a block of code periodically. How to use:
        ```
        cycler=Cycler(period=7)
        for t in range(1000):
            with cycler(t) as run, run:
                # code block to run periodically
                print(f"run at {t}")
        ```
        ATTENTION:
        The `with cycler (t) as run, run` style is critical, the second `run` is to trigger a nested with clause.
        - If the run condition is not met, then it will raise NotAtPeriod exception which will be caught by the outer
            with clause, and the code block will not run
        - If the run condition is met, then the code block will run

        Without the nested with clause, the period will not be checked against, and Cycler will raise an Exception

        This design is due to that __enter__ cannot skip the statement body: https://www.python.org/dev/peps/pep-0377/
        The proposal has been rejected

        :param period: period for a code block to run
        """
        assert period > 0, f"period must be > 0, but {period} is given"
        self._period = period
        self._last_run_at = start - period if run_at_start else start
        self._t = start
        self._entered = False
        self._conditions_checked = False

    @property
    def period(self):
        return self._period

    @property
    def last_run_at(self):
        return self._last_run_at

    def __call__(self, t: int):
        assert t >= 0, f"t must >=0, but {t} is given"
        self._t = t
        self._conditions_checked = False
        return self

    def __enter__(self):
        if self._entered:
            # this is the inner with
            self._conditions_checked = True
            if self._t - self._last_run_at < self._period:
                raise NotAtPeriod
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._conditions_checked:
            # the inner with never reached:
            raise Exception("The period has not been checked against. "
                            "Did you use 'with circulator(t) as run, run:' style?")
        if self._entered:
            # the inner with
            self._entered = False
            if exc_type is NotAtPeriod:
                # period not reached. did not run
                return True
            else:
                # clause ran already
                self._last_run_at = self._t
                # any other exception will be raised
