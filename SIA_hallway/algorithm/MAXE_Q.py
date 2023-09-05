import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class MAXE_Q_learning():
    """docstring for DQN"""
    def __init__(self, args, state_num, action_num):
        super(MAXE_Q_learning, self).__init__()
        self.args = args
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, action_num])
        self.lr = args.lr
        self.tau = 0.5
        self.eps = 1e-5

    def choose_action(self, state, epsilon):
        action_value = self.Q_value[state]
        # Boltzmann policy used to select action
        pi = np.exp(action_value / self.tau) / np.sum(np.exp(action_value / self.tau))
        log_pi = np.log(pi + self.eps)
        action = 0
        random_num = np.random.rand()
        for i in range(self.action_num):
            if random_num > np.sum(pi[:i]):
                action = i
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.action_num)
        return action, pi, log_pi

    def learn(self, state, action, reward, next_state):
        self.pi = np.exp(self.Q_value[next_state] / self.tau) / np.sum(np.exp(self.Q_value[next_state] / self.tau))
        self.entropy = np.sum(-self.pi * np.log(self.pi+self.eps))
        #self.Q_value[state, action] = reward + 0.99 * np.max(self.Q_value[next_state])
        self.Q_value[state, action] += self.lr * (reward + 0.01 * self.entropy +
                                                  0.99 * np.max(self.Q_value[next_state]) - self.Q_value[state, action])

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
