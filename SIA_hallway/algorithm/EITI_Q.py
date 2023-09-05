import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class EITI_Q_learning:
    """docstring for DQN"""
    def __init__(self, args, state_num, action_num):
        super(EITI_Q_learning, self).__init__()
        self.args = args
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, action_num])
        self.lr = args.lr

    def choose_action(self, state, epsilon):
        action_value = self.Q_value[state]
        max_action = np.where(action_value == np.max(action_value))
        action = max_action[0][np.random.randint(0, len(max_action))]
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, state, action, reward, next_state, intrisic_reward):
        self.Q_value[state, action] += self.lr * (reward + 0.1 * intrisic_reward +
                                                  0.99 * np.max(self.Q_value[next_state]) - self.Q_value[state, action])

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q