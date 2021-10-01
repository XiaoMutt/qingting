class LinearScheduler(object):
    def __init__(self, value_begin: float, value_end: float, descending_steps: int):
        """
        A Scheduler for shrinking the given value over time:
        - it's a linear interpolation from value_begin to value_end through descending_steps
        - after descending_steps, it remains at value_end
        :param value_begin:
        :param value_end:
        :param descending_steps:
        """
        self.value_begin = value_begin
        self.value_end = value_end
        self.descending_steps = descending_steps

    def value(self, t: int):
        """
        Update epsilon according to time step t
        :param t:
        :return: value at time step t
        """

        if t < self.descending_steps:
            return (self.descending_steps - t) / self.descending_steps * self.value_begin + \
                   t / self.descending_steps * self.value_end
        else:
            return self.value_end
