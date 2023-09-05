import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Q_learning():
    """docstring for DQN"""
    def __init__(self, state_num, action_num):
        super(Q_learning, self).__init__()
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, action_num])

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action_value = self.Q_value[state]
            max_action = np.where(action_value == np.max(action_value))
            action = max_action[0][np.random.randint(0, len(max_action))]
        else:
            action = np.random.randint(0, self.action_num)
        # if np.random.randn() <= episolon:
        #     action_value = self.Q_value[state]
        #     action_value = torch.tensor(action_value)
        #     p_action = F.gumbel_softmax(action_value)
        #     c = Categorical(p_action)
        #     action = c.sample().cpu().item()
        # else:
        #     action = np.random.randint(0, self.action_num)

        return action

    def learn(self, state, action, reward, next_state):
        self.Q_value[state, action] = reward + 0.99 * np.max(self.Q_value[next_state])

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q