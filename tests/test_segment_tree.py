from unittest import TestCase

import numpy as np

from core.ml.dqn.replay_buffer.segment_tree import SumSegmentTree, MinSegmentTree


class TestSegmentTree(TestCase):

    def test_sum_segment_tree(self):
        for _ in range(3):
            length = 1000
            oracle = np.random.random(length) * 100 - 50
            sst = SumSegmentTree(length)
            for i, e in enumerate(oracle):
                sst[i] = e

            for _ in range(1000):
                segment = sorted(np.random.randint(0, length, 2))
                self.assertAlmostEqual(sst.reduce(*segment), oracle[segment[0]:segment[1] + 1].sum())

            self.assertAlmostEqual(sst.reduce(3, -2), oracle[3: -1].sum())
            self.assertAlmostEqual(sst.reduce(6), oracle[6:].sum())

    def test_sum_segment_tree_max_idx0(self):
        sst = SumSegmentTree(4)
        sst[0] = -1
        sst[1] = 2
        sst[2] = 4
        sst[3] = 0
        self.assertEqual(sst.max_idx(-2), -1)
        self.assertEqual(sst.max_idx(-1), 0)
        self.assertEqual(sst.max_idx(0), 0)
        self.assertEqual(sst.max_idx(1), 1)
        self.assertEqual(sst.max_idx(2), 1)
        self.assertEqual(sst.max_idx(3), 1)
        self.assertEqual(sst.max_idx(4), 1)
        self.assertEqual(sst.max_idx(5), 3)
        self.assertEqual(sst.max_idx(6), 3)

    def test_sum_segment_tree_max_idx1(self):
        length = 1000
        oracle = np.random.randint(0, 1000, length)
        sst = SumSegmentTree(length)
        for i, e in enumerate(oracle):
            sst[i] = e

        for _ in range(1000):
            ceiling = np.random.random() * oracle.sum()
            sum_ = 0
            target = 0
            for i, e in enumerate(oracle):
                sum_ += e
                target = i
                if sum_ == ceiling:
                    break

                if sum_ > ceiling:
                    target -= 1
                    break
            self.assertEqual(sst.max_idx(ceiling), target)

    def test_min_segment_tree(self):
        for _ in range(3):
            length = 100000
            oracle = np.random.random(length) * 100 - 50
            mst = MinSegmentTree(length)
            for i, e in enumerate(oracle):
                mst[i] = e

            for _ in range(1000):
                segment = sorted(np.random.randint(0, length, 2))
                self.assertEqual(mst.reduce(*segment), oracle[segment[0]:segment[1] + 1].min())

            self.assertEqual(mst.reduce(3, -2), oracle[3: -1].min())
            self.assertEqual(mst.reduce(6), oracle[6:].min())
