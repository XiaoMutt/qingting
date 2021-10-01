import time
from collections import defaultdict


class Stopwatch(object):

    def __init__(self, enabled=False) -> None:
        self.enabled = enabled
        self.total_seconds = 0
        self.latest_start = 0
        self.num_calls = 0
        self.trie = defaultdict(lambda: Stopwatch(self.enabled))

    def __getattr__(self, item):
        return self.trie[item]

    def __getitem__(self, item):
        return self.trie[item]

    def __enter__(self):
        if self.enabled:
            self.latest_start = time.perf_counter()
            self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.total_seconds = time.perf_counter() - self.latest_start

    def __call__(self):
        total_seconds = self.total_seconds
        num_calls = self.num_calls
        messages = [None]
        for key, val in self.trie.items():
            t, n, msges = val()
            total_seconds += t
            num_calls += n
            for m in msges:
                messages.append(f'[{key}]' + m)

        messages[0] = f' total {total_seconds:.6f} seconds, {num_calls} calls, ' \
                      f'on average {total_seconds / num_calls:.6f} (sec/call)'
        return total_seconds, num_calls, messages

    def __str__(self):
        if self.enabled:
            _, _, messages = self()
            return '\n'.join(messages)
        else:
            return 'Not enabled'
