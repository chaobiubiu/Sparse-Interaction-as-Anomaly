import os
import sys
sys.path.append("..")
import argparse
import random
import datetime
import torch as th
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.MAXE_Q import MAXE_Q_learning
import time
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser("Experiments of MAXEntropyQ")
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
    args.max_steps = max_steps
    map_size = 7

    args.use_cuda = True
    args.device = "cuda:2" if args.use_cuda else "cpu"

    delta_epsilon = (args.epsilon_start - args.epsilon_finish) / args.epsilon_anneal_time

    rewards_agent = np.zeros([1, max_iteration, max_episode])
    for num_agent in [2]:
        print("The number of agents", num_agent)
        env = EnvGoObstacle(map_size, num_agent)

        log_path = 'runs/initial_two_rooms/' + 'n_agents_{}'.format(num_agent) + '/' + \
                   '0822night_MaxEntropy0.01_lr_{}__epsilon_anneal_{}_{}'.format(args.lr,
                                                                             args.epsilon_anneal_time,
                                                                             datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = SummaryWriter(log_path)

        Q_learning = [MAXE_Q_learning(args, map_size * map_size, 5) for _ in range(num_agent)]
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
                    epsilon = max(args.epsilon_finish, args.epsilon_start - delta_epsilon * (i * args.max_steps + count))
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        action, _, _ = Q_learning[index].choose_action(s, epsilon)
                        action_list.append(action)

                    reward, done, next_state = env.step(action_list)

                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0]*map_size+next_state[index][1]
                        rewards[index] += r
                        Q_learning[index].learn(s,a,r,s_)

                    state = next_state
                    count += 1

                    if count >= args.max_steps - 1:
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