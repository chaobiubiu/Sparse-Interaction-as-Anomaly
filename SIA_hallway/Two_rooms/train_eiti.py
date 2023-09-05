import os
import sys
sys.path.append("..")
import argparse
import datetime
import random
import torch as th
import numpy as np
from tqdm import tqdm
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.EITI_Q import EITI_Q_learning
import math
import time
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser("Experiments of EITI")
# Core training parameters
parser.add_argument("--lr", type=float, default=0.7, help="learning rate of q")

parser.add_argument("--epsilon_start", type=float, default=1.0)
parser.add_argument("--epsilon_finish", type=float, default=0.05)
parser.add_argument("--epsilon_anneal_time", type=int, default=10000)

args = parser.parse_args()


if __name__ == '__main__':
    max_episode = 500
    max_iteration = 10
    max_steps = 50
    args.max_steps = 50
    map_size = 7

    args.use_cuda = True
    args.device = "cuda:2" if args.use_cuda else "cpu"

    delta_epsilon = (args.epsilon_start - args.epsilon_finish) / args.epsilon_anneal_time

    rewards_agent = np.zeros([1, max_iteration, max_episode])
    for num_agent in [2]:
        print("The number of agents", num_agent)
        env = EnvGoObstacle(map_size, num_agent)

        log_path = 'runs/initial_two_rooms/' + 'n_agents_{}'.format(num_agent) + '/' + \
                   '0822night_EITI_lr_{}__epsilon_anneal_{}_{}'.format(args.lr, args.epsilon_anneal_time,
                                                                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = SummaryWriter(log_path)

        Q_learning = [EITI_Q_learning(args, map_size * map_size, 5) for _ in range(num_agent)]
        visited_num_sa = np.zeros([num_agent, map_size * map_size, 5, num_agent, map_size * map_size, 5, map_size * map_size])
        visited_num = np.zeros([map_size, map_size])
        for iter in range(max_iteration):
            seed = np.random.randint(0, 999999)
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)

            # Initialize Q with single-agent optimal Q
            for index in range(num_agent):
                if index == 0:
                    Q = np.load("li_Q_value1.npy")  # Agent_0 loads li_Q_value1, Agent_1 loads li_Q_value2
                elif index == 1:
                    Q = np.load("li_Q_value2.npy")
                else:
                    raise Exception("Currently consider only two agent setting.")
                Q_learning[index].set_Q_value(Q)

            all_steps = 0
            for i in tqdm(range(max_episode)):
                state = env.reset()
                count = 0
                done = False
                rewards = np.zeros(num_agent)

                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        visited_num[state[index][0]][state[index][1]] = visited_num[state[index][0]][state[index][1]] + 1
                        # epsilon = np.min([0.9, 0.7+(0.9-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                        epsilon = max(args.epsilon_finish, args.epsilon_start - delta_epsilon * (i * args.max_steps + count))

                        action = Q_learning[index].choose_action(s, epsilon)
                        action_list.append(action)

                    reward, done, next_state = env.step(action_list)

                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        a = action_list[index]
                        for other_index in range(num_agent):
                            if other_index != index:
                                s_other = state[other_index][0] * map_size + state[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * map_size + next_state[other_index][1]
                                visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] += 1

                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0]*map_size+next_state[index][1]
                        rewards[index] += r
                        r_in = 0
                        for other_index in range(num_agent):
                            if other_index != index:
                                s_other = state[other_index][0] * map_size + state[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * map_size + next_state[other_index][1]
                                if np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) == 0 or np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :]) == 0:
                                    r_temp = 0
                                else:
                                    r_temp = math.log( (visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] / np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) ) /
                                                       (np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, s_next_other]) / np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :])) )
                                r_in += r_temp
                        Q_learning[index].learn(s, a, r, s_, r_in)
                    state = next_state
                    count += 1

                    if count >= max_steps - 1:
                        done = True

                rewards_agent[0][iter][i] = np.mean(rewards)  # total reward calculation
                print("iter_{}_episode_{}_episodic_reward".format(iter, i), count, rewards[0], rewards[1])
                logger.add_scalar('iter_{}_episodic_reward'.format(iter), np.mean(rewards), i)
                logger.add_scalar('iter_{}_epsilon'.format(iter), epsilon, i)
                # print(i)
                all_steps += count
                # steps_mean[num_agent - 5, iter] = all_steps

                # Per episode, record the episodic step
                logger.add_scalar('iter_{}_step'.format(iter), count, i)