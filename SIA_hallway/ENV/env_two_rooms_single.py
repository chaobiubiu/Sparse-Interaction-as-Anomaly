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
        # 产生初始状态
        if self.num_agent == 1:
            inter_length = int(self.map_size // 2)
            for j in range(self.map_size):
                if j != inter_length:
                    self.obstacles[inter_length, j] = 1
            for i in range(self.num_agent):
                goal = [self.map_size - 1, self.map_size - 1]
                # goal = [self.map_size - 1, 0]
                self.goal.append(goal)
                # state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                # while state in self.goal or self.obstacles[state[0]][state[1]] == 1:
                #     # If randomly sampled initial states lie in the goals or obstacles, resample the initial locations.
                #     state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                state = [0, 0]
                # state = [0, self.map_size - 1]
                self.state.append(state)
        else:
            raise Exception("Only support 1 agent setting.")
        return self.state

    def get_env_info(self):
        pass

    def get_reward(self, state, action_list):
        reward = np.zeros(self.num_agent)     # Per step, agents receive 0 reward.
        next_state = copy.deepcopy(state)
        state = copy.deepcopy(state)
        for i in range(self.num_agent):
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
            if next_state[i] == self.goal[i]:
                reward[i] += 10     # If reach goal, receive +10

        # for i in range(self.num_agent):
        #     other_next_state = next_state[:i] + next_state[i + 1:]
        #     other_state = state[:i] + state[i + 1:]
        #     if next_state[i] == self.goal[i]:
        #         reward[i] += 10       # If reach goal, receive +10
        #     elif next_state[i][0] < 0 or next_state[i][0] > self.map_size - 1 or next_state[i][1] < 0 or \
        #             next_state[i][1] > self.map_size - 1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
        #         # If step outside the map's margin
        #         next_state[i] = state[i]
        #         reward[i] -= 0
        #     elif next_state[i] in other_next_state and next_state[i] == self.hall:
        #         # If collision occurs in the hall, current agent receives -1 reward
        #         next_state[i] = state[i]
        #         reward[i] -= 1
        #     elif next_state[i] in other_state and next_state[i] == self.hall:
        #         for j in range(len(other_state)):
        #             # A steps to B's pos, B steps to A's pos, then collision occurs and both agents stay still and receive -0.5 reward.
        #             if next_state[i] == other_state[j] and state[i] == other_next_state[j]:
        #                 next_state[i] = state[i]
        #                 reward[i] -= 1
        return reward, next_state

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

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 4))
        for i in range(self.map_size):
            for j in range(self.map_size):
                #if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i % 6 == 0:
                # 分第一组
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i % 6 == 1:
                # 第二组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i % 6 == 2:
                #第三组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i % 6 == 3:
                #第四组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i % 6 == 4:
                #第五组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            else:
                #第六组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0

        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        obs = self.get_global_obs()
        enlarge = 40
        henlarge = int(enlarge/2)
        qenlarge = int(enlarge/8)
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                # 红色方形agent及其红色圆形目标
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2*j+1) * henlarge, (2*i+1) * henlarge), henlarge, (0, 0, 255),-1)
                    #order = str(self.order[0].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 绿色方形agent及其绿色圆形目标
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 0), -1)
                    #order = str(self.order[1].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 蓝色方形agent及其蓝色圆形目标
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 0), -1)
                    #order = str(self.order[2].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 青色方形agent及其青色圆形目标
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 255, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 255, 0), -1)
                    #order = str(self.order[3].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 黄色方形agent及其黄色圆形目标
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 255), -1)
                    #order = str(self.order[4].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 粉色方形agent及其粉色圆形目标
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 255), -1)
                if self.obstacles[i][j] == 1:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                    #order = str(self.order[5].index(1) + 1)
                    #cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('image', new_obs)
        cv2.waitKey(600)