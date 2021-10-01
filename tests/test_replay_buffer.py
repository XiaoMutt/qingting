from unittest import TestCase

import numpy as np
import torch

from core.ml.dqn.replay_buffer.replay_buffer import ReplayBuffer


class TestReplayBuffer(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame_shape = (3, 5)
        cls.rb = ReplayBuffer(size=10, frame_num_per_state=4, frame_shape=cls.frame_shape)

    def test_0_add(self):

        with self.assertRaises(Exception):
            self.rb._get_encoded_state_at(8)

        for counter in range(0, 8):
            frame = np.full(self.frame_shape, counter)
            state = self.rb.add_frame(frame)
            self.assertTrue(torch.equal(state,
                                        torch.tensor([
                                            [np.full(self.frame_shape, max(0, counter + v)) for v in range(-3, 1)]
                                        ], dtype=torch.float32)))
            self.rb.add_feedback(0, 0, counter % 7 == 0)

        with self.assertRaises(Exception):
            self.rb._get_encoded_state_at(9)

        # frame 8
        frame = np.full(self.frame_shape, 8)
        state = self.rb.add_frame(frame)
        self.assertTrue(torch.equal(state, torch.tensor([
            [np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 0),
             frame
             ]
        ], dtype=torch.float32)))
        self.rb.add_feedback(0, 0, False)

        # frame 9
        frame = np.full(self.frame_shape, 9)
        state = self.rb.add_frame(frame)
        self.assertTrue(torch.equal(state, torch.tensor([
            [np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 8),
             frame
             ]
        ], dtype=torch.float32)))
        self.rb.add_feedback(0, 0, False)

        # frame 10
        frame = np.full(self.frame_shape, 10)
        state = self.rb.add_frame(frame)
        self.assertTrue(torch.equal(state, torch.tensor([
            [np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 8),
             np.full(self.frame_shape, 9),
             frame
             ]
        ], dtype=torch.float32)))
        self.rb.add_feedback(0, 0, False)

        frames = self.rb._get_encoded_state_at(10)
        self.assertTrue(np.array_equal(frames, np.array(
            [
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 8),
                np.full(self.frame_shape, 9),
                np.full(self.frame_shape, 10)
            ]
        )))

        frames = self.rb._get_encoded_state_at(9)
        self.assertTrue(np.array_equal(frames, np.array(
            [
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 8),
                np.full(self.frame_shape, 9)
            ]
        )))

        # fill until 19
        for counter in range(11, 20):
            frame = np.full(self.frame_shape, counter)
            state = self.rb.add_frame(frame)
            self.rb.add_feedback(0, 0, counter % 7 == 0)

        frames = self.rb._get_encoded_state_at(13)
        self.assertTrue(np.array_equal(frames, np.array(
            [
                np.full(self.frame_shape, 10),
                np.full(self.frame_shape, 11),
                np.full(self.frame_shape, 12),
                np.full(self.frame_shape, 13)
            ]
        )))

        frames = self.rb._get_encoded_state_at(14)
        self.assertTrue(np.array_equal(frames, np.array(
            [
                np.full(self.frame_shape, 11),
                np.full(self.frame_shape, 12),
                np.full(self.frame_shape, 13),
                np.full(self.frame_shape, 14)
            ]
        )))

        frames = self.rb._get_encoded_state_at(15)
        self.assertTrue(np.array_equal(frames, np.array(
            [
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 0),
                np.full(self.frame_shape, 15)
            ]
        )))

        # fill until 28
        for counter in range(20, 29):
            frame = np.full(self.frame_shape, counter)
            state = self.rb.add_frame(frame)
            self.rb.add_feedback(0, 0, counter % 7 == 0)

        with self.assertRaises(Exception):
            self.rb._get_encoded_state_at(29)

        with self.assertRaises(Exception):
            frames = self.rb._get_encoded_state_at(0)

        # frame 29
        frame = np.full(self.frame_shape, 29)
        state = self.rb.add_frame(frame)
        self.rb.add_feedback(0, 0, 29 % 7 == 0)

        frames = self.rb._get_encoded_state_at(29)
        self.assertTrue(np.array_equal(frames, np.array(
            [np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 0),
             np.full(self.frame_shape, 29)
             ])))
