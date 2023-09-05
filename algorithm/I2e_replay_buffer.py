import threading
import numpy as np
import random


class I2e_Buffer:
    def __init__(self, args):
        self.args = args
        # create the buffer to store info
        self.s = np.empty([self.args.buffer_size, self.args.max_steps, 1])
        self.a = np.empty([self.args.buffer_size, self.args.max_steps, 1])
        self.r = np.empty([self.args.buffer_size, self.args.max_steps, 1])
        self.s_ = np.empty([self.args.buffer_size, self.args.max_steps, 1])
        self.mask = np.empty([self.args.buffer_size, self.args.max_steps, 1])

        self.actual_length = 0
        self.index = 0

    def __len__(self):
        return self.actual_length

    def add(self, s, a, r, s_next, mask):
        self.actual_length = min(self.args.buffer_size, self.actual_length + 1)
        self.s[self.index] = np.expand_dims(s, axis=1)
        self.a[self.index] = np.expand_dims(a, axis=1)
        self.r[self.index] = np.expand_dims(r, axis=1)
        self.s_[self.index] = np.expand_dims(s_next, axis=1)
        self.mask[self.index] = np.expand_dims(mask, axis=1)
        self.index = (self.index + 1) % self.args.buffer_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        sample_list = [i for i in range(self.actual_length)]
        idxes = random.sample(sample_list, batch_size)
        episode_data = {}
        episode_data['s'] = self.s[idxes]
        episode_data['a'] = self.a[idxes]
        episode_data['r'] = self.r[idxes]
        episode_data['s_'] = self.s_[idxes]
        episode_data['mask'] = self.mask[idxes]
        return episode_data