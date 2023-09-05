import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import Counter
import cv2
import copy


class EnvGoObstacle(object):
    def __init__(self, map_size, num_agent):
        self.map_size = map_size
        self.num_agent = num_agent
        self.state = []
        self.goal  = []
        self.hall = [int(self.map_size // 2), int(self.map_size // 2)]
        self.occupancy = np.zeros((self.map_size, self.map_size))
        self.obstacles = np.zeros((self.map_size, self.map_size))

    def reset(self):
        self.state = []
        self.goal = []
        if self.num_agent == 2:
            inter_length = int(self.map_size // 2)
            for j in range(self.map_size):
                if j != inter_length:
                    self.obstacles[inter_length, j] = 1
            for i in range(self.num_agent):
                if i == 0:
                    goal = [self.map_size - 1, self.map_size - 1]
                else:
                    goal = [self.map_size - 1, 0]
                self.goal.append(goal)
                # state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                # while state in self.goal or self.obstacles[state[0]][state[1]] == 1:
                #     # If randomly sampled initial states lie in the goals or obstacles, resample the initial locations.
                #     state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                if i == 0:
                    state = [0, 0]
                else:
                    state = [0, self.map_size - 1]
                self.state.append(state)
        else:
            raise Exception("Only support 2 agents setting.")
        return self.state

    def get_env_info(self):
        pass

    def get_reward(self, state, action_list):
        reward = np.zeros(self.num_agent)     # Per step, agents receive 0 reward.
        next_state = copy.deepcopy(state)
        state = copy.deepcopy(state)
        for i in range(self.num_agent):
            if state[i] == self.goal[i]:
                continue
            if action_list[i] == 0:  # move right
                next_state[i][1] = state[i][1] + 1
            elif action_list[i] == 1:  # move left
                next_state[i][1] = state[i][1] - 1
            elif action_list[i] == 2:  # move up
                next_state[i][0] = state[i][0] - 1
            elif action_list[i] == 3:  # move down
                next_state[i][0] = state[i][0] + 1
            elif action_list[i] == 4:  # stay
                pass

        for i in range(self.num_agent):
            other_state = state[:i] + state[i+1:]
            if next_state[i][0] < 0 or next_state[i][0] > self.map_size - 1 or next_state[i][1] < 0 or \
                    next_state[i][1] > self.map_size - 1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
                # If step outside the map's margin
                next_state[i] = state[i]
                reward[i] -= 0
            else:
                temp_target = next_state[i]
                is_crowded = (self.hall in other_state)     # Whether there are other agents currently lie in the hall
                if is_crowded and temp_target == self.hall:
                    reward[i] -= 5
                    next_state[i] = state[i]        # If there is an agent already in the hall, stand still
                else:
                    state[i] = next_state[i]       # Otherwise walk into the target pos, update for sequent agents' panduan.

            # elif next_state[i] in other_next_state and next_state[i] == self.hall:
            #     # If collision occurs in the hall, current agent receives -1 reward
            #     next_state[i] = state[i]
            #     reward[i] -= 5
            # elif next_state[i] in other_state and next_state[i] == self.hall:
            #     for j in range(len(other_state)):
            #         # A steps to B's pos, B steps to A's pos, then collision occurs and both agents stay still and receive -0.5 reward.
            #         if next_state[i] == other_state[j] and state[i] == other_next_state[j]:
            #             next_state[i] = state[i]
            #             reward[i] -= 5
            if next_state[0] == self.goal[0] and next_state[1] == self.goal[1]:
                reward[i] += 10       # If reach goal, receive +10
        return reward, next_state

    # Maybe error.
    # def get_reward(self, state, action_list):
    #     reward = np.zeros(self.num_agent)     # Per step, agents receive 0 reward.
    #     next_state = copy.deepcopy(state)
    #     for i in range(self.num_agent):
    #         if state[i] == self.goal[i]:
    #             continue
    #         if action_list[i] == 0:  # move right
    #             next_state[i][1] = state[i][1] + 1
    #         elif action_list[i] == 1:  # move left
    #             next_state[i][1] = state[i][1] - 1
    #         elif action_list[i] == 2:  # move up
    #             next_state[i][0] = state[i][0] - 1
    #         elif action_list[i] == 3:  # move down
    #             next_state[i][0] = state[i][0] + 1
    #         elif action_list[i] == 4:  # stay
    #             pass
    #
    #     for i in range(self.num_agent):
    #         other_next_state = next_state[:i] + next_state[i + 1:]
    #         other_state = state[:i] + state[i + 1:]
    #
    #         if next_state[i][0] < 0 or next_state[i][0] > self.map_size - 1 or next_state[i][1] < 0 or \
    #                 next_state[i][1] > self.map_size - 1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
    #             # If step outside the map's margin
    #             next_state[i] = state[i]
    #             reward[i] -= 0
    #         elif next_state[i] in other_next_state and next_state[i] == self.hall:
    #             # If collision occurs in the hall, current agent receives -1 reward
    #             next_state[i] = state[i]
    #             reward[i] -= 5
    #         elif next_state[i] in other_state and next_state[i] == self.hall:
    #             for j in range(len(other_state)):
    #                 # A steps to B's pos, B steps to A's pos, then collision occurs and both agents stay still and receive -0.5 reward.
    #                 if next_state[i] == other_state[j] and state[i] == other_next_state[j]:
    #                     next_state[i] = state[i]
    #                     reward[i] -= 5
    #         if next_state[0] == self.goal[0] and next_state[1] == self.goal[1]:
    #             reward[i] += 10       # If reach goal, receive +10
    #     return reward, next_state

    def step(self, action_list):
        reward, next_state = self.get_reward(self.state, action_list)
        done = True
        for i in range(self.num_agent):
            if next_state[i] != self.goal[i]:
                done = False
        self.state = next_state
        return reward, done, self.state

    def sqr_dist(self, pos1, pos2):
        return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])
